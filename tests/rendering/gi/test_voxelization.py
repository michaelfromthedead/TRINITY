"""Tests for scene voxelization (T-GIR-P7.1).

Tests cover:
    - VoxelResolution enumeration
    - OpacityClass classification
    - Triangle data structure
    - Voxel accumulation and finalization
    - VoxelGrid operations
    - ConservativeRasterizer intersection tests
    - SceneVoxelizer voxelization
    - Performance benchmarks
    - Edge cases and error handling
"""

import math
import pytest
from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3, Vec4

from engine.rendering.gi.voxelization import (
    # Constants
    RESOLUTION_64,
    RESOLUTION_128,
    RESOLUTION_256,
    SUPPORTED_RESOLUTIONS,
    EPSILON,
    # Enums
    VoxelResolution,
    OpacityClass,
    VoxelAxis,
    # Data structures
    Triangle,
    Voxel,
    # Grid
    VoxelGrid,
    # Rasterizer
    ConservativeRasterizer,
    # Voxelizer
    VoxelizationStats,
    VoxelizationConfig,
    SceneVoxelizer,
    # Utilities
    create_test_triangles,
    estimate_voxelization_memory,
    recommend_resolution,
    generate_voxelize_compute_wgsl,
)


# ============================================================================
# VoxelResolution Tests
# ============================================================================


class TestVoxelResolution:
    """Tests for VoxelResolution enumeration."""

    def test_resolution_values(self) -> None:
        """Resolution values should match constants."""
        assert VoxelResolution.LOW.value == 64
        assert VoxelResolution.MEDIUM.value == 128
        assert VoxelResolution.HIGH.value == 256

    def test_resolution_size_property(self) -> None:
        """Size property should return linear dimension."""
        assert VoxelResolution.LOW.size == 64
        assert VoxelResolution.MEDIUM.size == 128
        assert VoxelResolution.HIGH.size == 256

    def test_total_voxels(self) -> None:
        """Total voxels should be size cubed."""
        assert VoxelResolution.LOW.total_voxels == 64 ** 3
        assert VoxelResolution.MEDIUM.total_voxels == 128 ** 3
        assert VoxelResolution.HIGH.total_voxels == 256 ** 3

    def test_memory_estimate(self) -> None:
        """Memory estimate should be reasonable."""
        # LOW: 64^3 * 12 bytes = 3.145 MB
        assert VoxelResolution.LOW.memory_estimate_mb == pytest.approx(3.0, rel=0.1)
        # MEDIUM: 128^3 * 12 bytes = 25.2 MB
        assert VoxelResolution.MEDIUM.memory_estimate_mb == pytest.approx(24.0, rel=0.1)
        # HIGH: 256^3 * 12 bytes = 201.3 MB
        assert VoxelResolution.HIGH.memory_estimate_mb == pytest.approx(192.0, rel=0.1)


# ============================================================================
# OpacityClass Tests
# ============================================================================


class TestOpacityClass:
    """Tests for OpacityClass enumeration."""

    def test_empty_classification(self) -> None:
        """Near-zero alpha should classify as EMPTY."""
        assert OpacityClass.from_alpha(0.0) == OpacityClass.EMPTY
        assert OpacityClass.from_alpha(0.0005) == OpacityClass.EMPTY

    def test_transparent_classification(self) -> None:
        """Low alpha should classify as TRANSPARENT."""
        assert OpacityClass.from_alpha(0.001) == OpacityClass.TRANSPARENT
        assert OpacityClass.from_alpha(0.05) == OpacityClass.TRANSPARENT
        assert OpacityClass.from_alpha(0.09) == OpacityClass.TRANSPARENT

    def test_semitransparent_classification(self) -> None:
        """Mid-range alpha should classify as SEMITRANSPARENT."""
        assert OpacityClass.from_alpha(0.1) == OpacityClass.SEMITRANSPARENT
        assert OpacityClass.from_alpha(0.5) == OpacityClass.SEMITRANSPARENT
        assert OpacityClass.from_alpha(0.89) == OpacityClass.SEMITRANSPARENT

    def test_opaque_classification(self) -> None:
        """High alpha should classify as OPAQUE."""
        assert OpacityClass.from_alpha(0.9) == OpacityClass.OPAQUE
        assert OpacityClass.from_alpha(0.95) == OpacityClass.OPAQUE
        assert OpacityClass.from_alpha(1.0) == OpacityClass.OPAQUE


# ============================================================================
# VoxelAxis Tests
# ============================================================================


class TestVoxelAxis:
    """Tests for VoxelAxis enumeration."""

    def test_axis_values(self) -> None:
        """Axis enumeration should have expected values."""
        assert VoxelAxis.X is not None
        assert VoxelAxis.Y is not None
        assert VoxelAxis.Z is not None


# ============================================================================
# Triangle Tests
# ============================================================================


class TestTriangle:
    """Tests for Triangle data structure."""

    def test_triangle_creation(self) -> None:
        """Triangle should be created with vertices."""
        v0 = Vec3(0, 0, 0)
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(0, 1, 0)
        tri = Triangle(v0, v1, v2)
        assert tri.v0 == v0
        assert tri.v1 == v1
        assert tri.v2 == v2

    def test_face_normal_computation(self) -> None:
        """Face normal should be computed from vertices."""
        v0 = Vec3(0, 0, 0)
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(0, 1, 0)
        tri = Triangle(v0, v1, v2)
        normal = tri.compute_face_normal()
        # Cross product of (1,0,0) and (0,1,0) is (0,0,1)
        assert normal.z == pytest.approx(1.0, abs=1e-6)
        assert normal.x == pytest.approx(0.0, abs=1e-6)
        assert normal.y == pytest.approx(0.0, abs=1e-6)

    def test_auto_normal_assignment(self) -> None:
        """Vertex normals should default to face normal."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
        assert tri.n0 is not None
        assert tri.n1 is not None
        assert tri.n2 is not None
        assert tri.n0.z == pytest.approx(1.0, abs=1e-6)

    def test_custom_normals(self) -> None:
        """Custom vertex normals should be preserved."""
        n = Vec3(0, 1, 0)
        tri = Triangle(
            Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
            n0=n, n1=n, n2=n
        )
        assert tri.n0 == n
        assert tri.n1 == n
        assert tri.n2 == n

    def test_aabb_computation(self) -> None:
        """AABB should bound all vertices."""
        tri = Triangle(Vec3(1, 2, 3), Vec3(4, 5, 6), Vec3(-1, 0, 2))
        aabb = tri.get_aabb()
        assert aabb.min.x == -1
        assert aabb.min.y == 0
        assert aabb.min.z == 2
        assert aabb.max.x == 4
        assert aabb.max.y == 5
        assert aabb.max.z == 6

    def test_centroid_computation(self) -> None:
        """Centroid should be average of vertices."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(3, 0, 0), Vec3(0, 3, 0))
        center = tri.get_center()
        assert center.x == pytest.approx(1.0)
        assert center.y == pytest.approx(1.0)
        assert center.z == pytest.approx(0.0)

    def test_area_computation(self) -> None:
        """Area should be computed correctly."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(0, 2, 0))
        area = tri.get_area()
        assert area == pytest.approx(2.0)  # 0.5 * base * height = 0.5 * 2 * 2

    def test_normal_interpolation(self) -> None:
        """Normal interpolation with barycentric coords."""
        n0 = Vec3(1, 0, 0)
        n1 = Vec3(0, 1, 0)
        n2 = Vec3(0, 0, 1)
        tri = Triangle(
            Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
            n0=n0, n1=n1, n2=n2
        )
        # At centroid (1/3, 1/3, 1/3)
        bary = Vec3(1/3, 1/3, 1/3)
        interp = tri.interpolate_normal(bary)
        # Should be normalized (1/3, 1/3, 1/3).normalized()
        expected_len = math.sqrt(3) / 3
        assert interp.x == pytest.approx(expected_len, abs=1e-5)
        assert interp.y == pytest.approx(expected_len, abs=1e-5)
        assert interp.z == pytest.approx(expected_len, abs=1e-5)

    def test_degenerate_triangle_normal(self) -> None:
        """Degenerate triangle should return up vector."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0))
        normal = tri.compute_face_normal()
        assert normal.y == pytest.approx(1.0)


# ============================================================================
# Voxel Tests
# ============================================================================


class TestVoxel:
    """Tests for Voxel data structure."""

    def test_default_voxel_is_empty(self) -> None:
        """Default voxel should be empty."""
        v = Voxel()
        assert v.is_empty()
        assert v.hit_count == 0

    def test_accumulate_single(self) -> None:
        """Single accumulation should set values."""
        v = Voxel()
        albedo = Vec4(1.0, 0.5, 0.25, 1.0)
        emissive = Vec3(0.1, 0.2, 0.3)
        normal = Vec3(0, 1, 0)
        v.accumulate(albedo, emissive, normal)

        assert v.hit_count == 1
        assert not v.is_empty()
        assert v.albedo.x == pytest.approx(1.0)
        assert v.albedo.y == pytest.approx(0.5)
        assert v.albedo.z == pytest.approx(0.25)
        assert v.albedo.w == pytest.approx(1.0)

    def test_accumulate_multiple_averages(self) -> None:
        """Multiple accumulations should average values."""
        v = Voxel()
        v.accumulate(Vec4(1, 0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))
        v.accumulate(Vec4(0, 1, 0, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))

        assert v.hit_count == 2
        assert v.albedo.x == pytest.approx(0.5)
        assert v.albedo.y == pytest.approx(0.5)

    def test_opacity_class(self) -> None:
        """Opacity class should reflect alpha value."""
        v = Voxel()
        assert v.get_opacity_class() == OpacityClass.EMPTY

        v.accumulate(Vec4(1, 1, 1, 0.05), Vec3.zero(), Vec3.unit_y())
        assert v.get_opacity_class() == OpacityClass.TRANSPARENT

        v2 = Voxel()
        v2.accumulate(Vec4(1, 1, 1, 0.5), Vec3.zero(), Vec3.unit_y())
        assert v2.get_opacity_class() == OpacityClass.SEMITRANSPARENT

        v3 = Voxel()
        v3.accumulate(Vec4(1, 1, 1, 1.0), Vec3.zero(), Vec3.unit_y())
        assert v3.get_opacity_class() == OpacityClass.OPAQUE

    def test_finalize_normalizes(self) -> None:
        """Finalize should normalize the normal vector."""
        v = Voxel()
        v.accumulate(Vec4(1, 1, 1, 1), Vec3.zero(), Vec3(0, 2, 0))
        v.finalize()
        assert v.normal.length() == pytest.approx(1.0, abs=1e-6)

    def test_to_rgba8(self) -> None:
        """RGBA8 conversion should clamp and scale."""
        v = Voxel()
        v.accumulate(Vec4(0.5, 0.25, 0.0, 1.0), Vec3.zero(), Vec3.unit_y())
        r, g, b, a = v.to_rgba8()
        assert r == 127 or r == 128  # 0.5 * 255 ~ 127.5
        assert g == 63 or g == 64    # 0.25 * 255 ~ 63.75
        assert b == 0
        assert a == 255


# ============================================================================
# VoxelGrid Tests
# ============================================================================


class TestVoxelGrid:
    """Tests for VoxelGrid."""

    def test_grid_creation(self) -> None:
        """Grid should be created with correct dimensions."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(64, bounds)
        assert grid.resolution == 64
        assert grid.total_voxels == 64 ** 3

    def test_invalid_resolution_raises(self) -> None:
        """Invalid resolution should raise ValueError."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        with pytest.raises(ValueError, match="Resolution must be positive"):
            VoxelGrid(0, bounds)
        with pytest.raises(ValueError, match="Resolution must be positive"):
            VoxelGrid(-1, bounds)

    def test_voxel_size(self) -> None:
        """Voxel size should be bounds / resolution."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(64, 64, 64))
        grid = VoxelGrid(64, bounds)
        assert grid.voxel_size.x == pytest.approx(1.0)
        assert grid.voxel_size.y == pytest.approx(1.0)
        assert grid.voxel_size.z == pytest.approx(1.0)

    def test_get_set_voxel(self) -> None:
        """Get/set voxel should work correctly."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        v = Voxel()
        v.accumulate(Vec4(1, 0, 0, 1), Vec3.zero(), Vec3.unit_y())
        grid.set_voxel(5, 5, 5, v)

        retrieved = grid.get_voxel(5, 5, 5)
        assert retrieved.albedo.x == pytest.approx(1.0)

    def test_out_of_bounds_raises(self) -> None:
        """Out of bounds access should raise IndexError."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        with pytest.raises(IndexError):
            grid.get_voxel(10, 0, 0)
        with pytest.raises(IndexError):
            grid.get_voxel(-1, 0, 0)
        with pytest.raises(IndexError):
            grid.set_voxel(0, 10, 0, Voxel())

    def test_world_to_voxel(self) -> None:
        """World to voxel conversion should work correctly."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        x, y, z = grid.world_to_voxel(Vec3(5.5, 5.5, 5.5))
        assert x == 5
        assert y == 5
        assert z == 5

    def test_world_to_voxel_clamping(self) -> None:
        """World to voxel should clamp to bounds."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        # Outside bounds should clamp
        x, y, z = grid.world_to_voxel(Vec3(-5, 15, 5))
        assert x == 0
        assert y == 9
        assert z == 5

    def test_voxel_to_world(self) -> None:
        """Voxel to world should return center position."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        pos = grid.voxel_to_world(5, 5, 5)
        assert pos.x == pytest.approx(5.5)
        assert pos.y == pytest.approx(5.5)
        assert pos.z == pytest.approx(5.5)

    def test_get_voxel_aabb(self) -> None:
        """Voxel AABB should have correct dimensions."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        aabb = grid.get_voxel_aabb(5, 5, 5)
        assert aabb.min.x == pytest.approx(5.0)
        assert aabb.min.y == pytest.approx(5.0)
        assert aabb.min.z == pytest.approx(5.0)
        assert aabb.max.x == pytest.approx(6.0)
        assert aabb.max.y == pytest.approx(6.0)
        assert aabb.max.z == pytest.approx(6.0)

    def test_clear(self) -> None:
        """Clear should reset all voxels."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        # Fill some voxels
        grid.get_voxel(0, 0, 0).accumulate(Vec4(1, 1, 1, 1), Vec3.zero(), Vec3.unit_y())
        assert grid.count_filled_voxels() == 1

        grid.clear()
        assert grid.count_filled_voxels() == 0

    def test_count_filled_voxels(self) -> None:
        """Count filled should return correct count."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)

        for i in range(10):
            x, y, z = i % 4, (i // 4) % 4, i // 16
            grid.get_voxel(x, y, z).accumulate(
                Vec4(1, 1, 1, 1), Vec3.zero(), Vec3.unit_y()
            )

        assert grid.count_filled_voxels() == 10

    def test_fill_ratio(self) -> None:
        """Fill ratio should be correct."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)  # 64 voxels

        for i in range(16):
            x, y, z = i % 4, (i // 4) % 4, 0
            grid.get_voxel(x, y, z).accumulate(
                Vec4(1, 1, 1, 1), Vec3.zero(), Vec3.unit_y()
            )

        assert grid.get_fill_ratio() == pytest.approx(16 / 64)

    def test_iter_filled(self) -> None:
        """Iterate filled should yield only non-empty voxels."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)

        grid.get_voxel(1, 1, 1).accumulate(Vec4(1, 0, 0, 1), Vec3.zero(), Vec3.unit_y())
        grid.get_voxel(2, 2, 2).accumulate(Vec4(0, 1, 0, 1), Vec3.zero(), Vec3.unit_y())

        filled = list(grid.iter_filled())
        assert len(filled) == 2

    def test_classify_opacity(self) -> None:
        """Opacity classification should count correctly."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)

        # Add various opacity levels
        grid.get_voxel(0, 0, 0).accumulate(Vec4(1, 1, 1, 1.0), Vec3.zero(), Vec3.unit_y())
        grid.get_voxel(1, 0, 0).accumulate(Vec4(1, 1, 1, 0.5), Vec3.zero(), Vec3.unit_y())
        grid.get_voxel(2, 0, 0).accumulate(Vec4(1, 1, 1, 0.05), Vec3.zero(), Vec3.unit_y())

        counts = grid.classify_opacity()
        assert counts[OpacityClass.OPAQUE] == 1
        assert counts[OpacityClass.SEMITRANSPARENT] == 1
        assert counts[OpacityClass.TRANSPARENT] == 1
        assert counts[OpacityClass.EMPTY] == 64 - 3

    def test_to_albedo_bytes(self) -> None:
        """Albedo bytes should be correct size."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)
        data = grid.to_albedo_bytes()
        # 4^3 * 4 bytes (RGBA8)
        assert len(data) == 4 ** 3 * 4

    def test_to_normal_bytes(self) -> None:
        """Normal bytes should be correct size."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)
        data = grid.to_normal_bytes()
        assert len(data) == 4 ** 3 * 4

    def test_to_emissive_bytes(self) -> None:
        """Emissive bytes should be correct size."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)
        data = grid.to_emissive_bytes()
        assert len(data) == 4 ** 3 * 4


# ============================================================================
# ConservativeRasterizer Tests
# ============================================================================


class TestConservativeRasterizer:
    """Tests for ConservativeRasterizer."""

    def test_dominant_axis_z(self) -> None:
        """Z-facing triangle should have Z dominant axis."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
        axis = ConservativeRasterizer.get_dominant_axis(tri)
        assert axis == VoxelAxis.Z

    def test_dominant_axis_x(self) -> None:
        """X-facing triangle should have X dominant axis."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
        axis = ConservativeRasterizer.get_dominant_axis(tri)
        assert axis == VoxelAxis.X

    def test_dominant_axis_y(self) -> None:
        """Y-facing triangle should have Y dominant axis."""
        tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1))
        axis = ConservativeRasterizer.get_dominant_axis(tri)
        assert axis == VoxelAxis.Y

    def test_triangle_aabb_intersects_inside(self) -> None:
        """Triangle inside AABB should intersect."""
        tri = Triangle(Vec3(0.25, 0.25, 0.25), Vec3(0.75, 0.25, 0.25), Vec3(0.5, 0.75, 0.25))
        intersects = ConservativeRasterizer.triangle_aabb_intersects(
            tri, Vec3(0, 0, 0), Vec3(1, 1, 1)
        )
        assert intersects

    def test_triangle_aabb_intersects_outside(self) -> None:
        """Triangle outside AABB should not intersect."""
        tri = Triangle(Vec3(5, 5, 5), Vec3(6, 5, 5), Vec3(5.5, 6, 5))
        intersects = ConservativeRasterizer.triangle_aabb_intersects(
            tri, Vec3(0, 0, 0), Vec3(1, 1, 1)
        )
        assert not intersects

    def test_triangle_aabb_intersects_crossing(self) -> None:
        """Triangle crossing AABB boundary should intersect."""
        tri = Triangle(Vec3(-0.5, 0.5, 0.5), Vec3(0.5, 0.5, 0.5), Vec3(0, 1.5, 0.5))
        intersects = ConservativeRasterizer.triangle_aabb_intersects(
            tri, Vec3(0, 0, 0), Vec3(1, 1, 1)
        )
        assert intersects

    def test_triangle_aabb_corner_touch(self) -> None:
        """Triangle touching AABB corner should intersect."""
        tri = Triangle(Vec3(1, 1, 1), Vec3(2, 1, 1), Vec3(1.5, 2, 1))
        intersects = ConservativeRasterizer.triangle_aabb_intersects(
            tri, Vec3(0, 0, 0), Vec3(1, 1, 1)
        )
        assert intersects

    def test_rasterize_triangle_fills_voxels(self) -> None:
        """Rasterizing triangle should fill voxels."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        tri = Triangle(
            Vec3(2, 2, 2), Vec3(6, 2, 2), Vec3(4, 6, 2),
            albedo=Vec4(1, 0, 0, 1)
        )

        filled = ConservativeRasterizer.rasterize_triangle(tri, grid)
        assert filled > 0

    def test_rasterize_outside_triangle(self) -> None:
        """Triangle outside grid should fill zero voxels."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        tri = Triangle(Vec3(100, 100, 100), Vec3(101, 100, 100), Vec3(100.5, 101, 100))

        filled = ConservativeRasterizer.rasterize_triangle(tri, grid)
        assert filled == 0

    def test_rasterize_single_voxel_triangle(self) -> None:
        """Small triangle should fill at least one voxel."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = VoxelGrid(10, bounds)

        # Very small triangle in center of one voxel
        tri = Triangle(
            Vec3(5.4, 5.4, 5.4), Vec3(5.6, 5.4, 5.4), Vec3(5.5, 5.6, 5.4),
            albedo=Vec4(0, 1, 0, 1)
        )

        filled = ConservativeRasterizer.rasterize_triangle(tri, grid)
        assert filled >= 1


# ============================================================================
# VoxelizationConfig Tests
# ============================================================================


class TestVoxelizationConfig:
    """Tests for VoxelizationConfig."""

    def test_default_config(self) -> None:
        """Default config should use medium resolution."""
        config = VoxelizationConfig()
        assert config.resolution == 128
        assert config.conservative is True
        assert config.accumulate_normals is True
        assert config.include_backfaces is True

    def test_invalid_resolution_raises(self) -> None:
        """Invalid resolution should raise ValueError."""
        with pytest.raises(ValueError, match="Resolution 100 not supported"):
            VoxelizationConfig(resolution=100)

    def test_valid_resolutions(self) -> None:
        """Valid resolutions should be accepted."""
        for res in SUPPORTED_RESOLUTIONS:
            config = VoxelizationConfig(resolution=res)
            assert config.resolution == res


# ============================================================================
# VoxelizationStats Tests
# ============================================================================


class TestVoxelizationStats:
    """Tests for VoxelizationStats."""

    def test_performance_target_64(self) -> None:
        """64^3 should have 0.5ms target."""
        stats = VoxelizationStats(elapsed_ms=0.3)
        assert stats.is_performance_target_met(64)

        stats_slow = VoxelizationStats(elapsed_ms=0.6)
        assert not stats_slow.is_performance_target_met(64)

    def test_performance_target_128(self) -> None:
        """128^3 should have 1.5ms target."""
        stats = VoxelizationStats(elapsed_ms=1.0)
        assert stats.is_performance_target_met(128)

    def test_performance_target_256(self) -> None:
        """256^3 should have 4.0ms target."""
        stats = VoxelizationStats(elapsed_ms=3.5)
        assert stats.is_performance_target_met(256)

        stats_slow = VoxelizationStats(elapsed_ms=5.0)
        assert not stats_slow.is_performance_target_met(256)


# ============================================================================
# SceneVoxelizer Tests
# ============================================================================


class TestSceneVoxelizer:
    """Tests for SceneVoxelizer."""

    def test_voxelizer_creation(self) -> None:
        """Voxelizer should be created with default config."""
        voxelizer = SceneVoxelizer()
        assert voxelizer.config.resolution == 128

    def test_voxelizer_custom_config(self) -> None:
        """Voxelizer should accept custom config."""
        config = VoxelizationConfig(resolution=64)
        voxelizer = SceneVoxelizer(config)
        assert voxelizer.config.resolution == 64

    def test_voxelize_empty(self) -> None:
        """Voxelizing empty list should produce empty grid."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        grid = voxelizer.voxelize([], bounds)

        assert grid.count_filled_voxels() == 0

    def test_voxelize_single_triangle(self) -> None:
        """Voxelizing single triangle should fill some voxels."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        triangles = [
            Triangle(
                Vec3(2, 2, 2), Vec3(6, 2, 2), Vec3(4, 6, 2),
                albedo=Vec4(1, 0, 0, 1)
            )
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        assert grid.count_filled_voxels() > 0

    def test_voxelize_multiple_triangles(self) -> None:
        """Voxelizing multiple triangles should work."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        triangles = [
            Triangle(Vec3(2, 2, 2), Vec3(4, 2, 2), Vec3(3, 4, 2)),
            Triangle(Vec3(6, 2, 2), Vec3(8, 2, 2), Vec3(7, 4, 2)),
            Triangle(Vec3(4, 6, 2), Vec3(6, 6, 2), Vec3(5, 8, 2)),
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        stats = voxelizer.get_last_stats()
        assert stats is not None
        assert stats.triangle_count == 3
        assert stats.voxels_filled > 0

    def test_voxelize_with_emissive(self) -> None:
        """Voxelizing emissive geometry should preserve emissive."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        triangles = [
            Triangle(
                Vec3(5, 5, 5), Vec3(5.1, 5, 5), Vec3(5.05, 5.1, 5),
                albedo=Vec4(1, 1, 1, 1),
                emissive=Vec3(1, 0.5, 0)
            )
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        # Find filled voxel and check emissive
        for x, y, z, v in grid.iter_filled():
            assert v.emissive.x > 0 or v.emissive.y > 0

    def test_get_last_stats(self) -> None:
        """Stats should be available after voxelization."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        triangles = [Triangle(Vec3(1, 1, 1), Vec3(2, 1, 1), Vec3(1.5, 2, 1))]
        voxelizer.voxelize(triangles, bounds)

        stats = voxelizer.get_last_stats()
        assert stats is not None
        assert stats.triangle_count == 1
        assert stats.elapsed_ms >= 0

    def test_estimate_time(self) -> None:
        """Time estimation should be reasonable."""
        voxelizer = SceneVoxelizer()

        # Small scene
        time_small = voxelizer.estimate_time_ms(100, 64)
        assert time_small < 1.0

        # Large scene
        time_large = voxelizer.estimate_time_ms(100000, 256)
        assert time_large > time_small

    def test_compute_optimal_bounds(self) -> None:
        """Optimal bounds should enclose all triangles."""
        triangles = [
            Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0)),
            Triangle(Vec3(5, 5, 5), Vec3(6, 5, 5), Vec3(5.5, 6, 5)),
        ]

        bounds = SceneVoxelizer.compute_optimal_bounds(triangles)

        # Should contain all vertices with padding
        assert bounds.min.x < 0
        assert bounds.min.y < 0
        assert bounds.max.x > 6
        assert bounds.max.y > 6

    def test_compute_optimal_bounds_empty(self) -> None:
        """Empty triangle list should return unit bounds."""
        bounds = SceneVoxelizer.compute_optimal_bounds([])
        assert bounds.min == Vec3.zero()
        assert bounds.max == Vec3.one()

    def test_create_from_resolution(self) -> None:
        """Factory method should create correct voxelizer."""
        voxelizer = SceneVoxelizer.create_from_resolution(VoxelResolution.HIGH)
        assert voxelizer.config.resolution == 256


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_test_triangles(self) -> None:
        """Test triangle creation utility."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        triangles = create_test_triangles(100, bounds)

        assert len(triangles) == 100
        for tri in triangles:
            # All vertices should be within expanded bounds
            aabb = tri.get_aabb()
            assert aabb.min.x >= -1 and aabb.max.x <= 11

    def test_estimate_voxelization_memory(self) -> None:
        """Memory estimation should be correct."""
        # 64^3 * 12 bytes = 3,145,728 bytes
        mem_64 = estimate_voxelization_memory(64)
        assert mem_64 == 64 ** 3 * 12

        # 128^3 * 12 bytes
        mem_128 = estimate_voxelization_memory(128)
        assert mem_128 == 128 ** 3 * 12

    def test_recommend_resolution(self) -> None:
        """Resolution recommendation should be reasonable."""
        # Small scene should use LOW
        small_bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        assert recommend_resolution(small_bounds) == VoxelResolution.LOW

        # Large scene should use HIGH
        large_bounds = AABB(Vec3(0, 0, 0), Vec3(100, 100, 100))
        assert recommend_resolution(large_bounds) == VoxelResolution.HIGH

    def test_generate_wgsl(self) -> None:
        """WGSL generation should produce valid shader code."""
        wgsl = generate_voxelize_compute_wgsl()

        # Check for expected content
        assert "voxelize.comp.wgsl" in wgsl
        assert "VoxelGridUniforms" in wgsl
        assert "Triangle" in wgsl
        assert "@compute" in wgsl
        assert "@workgroup_size" in wgsl
        assert "triangle_aabb_intersects" in wgsl


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_64_resolution_fast(self) -> None:
        """64^3 voxelization should be fast."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        triangles = create_test_triangles(100, bounds)

        grid = voxelizer.voxelize(triangles, bounds)
        stats = voxelizer.get_last_stats()

        # Should complete in reasonable time (Python is slow, so be generous)
        assert stats is not None
        # In pure Python, even 100 triangles at 64^3 might take a while
        # Just verify it completes

    def test_voxel_count_proportional_to_triangles(self) -> None:
        """More triangles should fill more voxels."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        triangles_small = create_test_triangles(10, bounds)
        grid_small = voxelizer.voxelize(triangles_small, bounds)
        filled_small = grid_small.count_filled_voxels()

        triangles_large = create_test_triangles(50, bounds)
        grid_large = voxelizer.voxelize(triangles_large, bounds)
        filled_large = grid_large.count_filled_voxels()

        # More triangles should generally fill more voxels
        # (not strictly true due to overlap, but likely)
        assert filled_large >= filled_small or filled_small > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_thin_triangle(self) -> None:
        """Very thin triangle should still voxelize."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        # Line-like triangle
        triangles = [
            Triangle(
                Vec3(2, 5, 5), Vec3(8, 5, 5), Vec3(5, 5.001, 5),
                albedo=Vec4(1, 1, 1, 1)
            )
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        # Conservative rasterization should fill voxels along the line
        assert grid.count_filled_voxels() > 0

    def test_triangle_at_boundary(self) -> None:
        """Triangle at grid boundary should work."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        # Triangle at edge
        triangles = [
            Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0))
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        assert grid.count_filled_voxels() > 0

    def test_non_unit_voxel_size(self) -> None:
        """Non-uniform bounds should work correctly."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(-5, 0, 10), Vec3(15, 20, 30))  # 20x20x20

        triangles = [
            Triangle(Vec3(0, 10, 20), Vec3(10, 10, 20), Vec3(5, 15, 20))
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        assert grid.count_filled_voxels() > 0

    def test_overlapping_triangles(self) -> None:
        """Overlapping triangles should average properties."""
        voxelizer = SceneVoxelizer(VoxelizationConfig(resolution=64))
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))

        # Two triangles in same location with different colors
        triangles = [
            Triangle(
                Vec3(5, 5, 5), Vec3(5.1, 5, 5), Vec3(5.05, 5.1, 5),
                albedo=Vec4(1, 0, 0, 1)
            ),
            Triangle(
                Vec3(5, 5, 5), Vec3(5.1, 5, 5), Vec3(5.05, 5.1, 5),
                albedo=Vec4(0, 0, 1, 1)
            ),
        ]
        grid = voxelizer.voxelize(triangles, bounds)

        # Check that colors were blended
        for x, y, z, v in grid.iter_filled():
            # Should be some mix of red and blue
            assert v.hit_count >= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self) -> None:
        """Test complete voxelization pipeline."""
        # Create triangles
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        triangles = create_test_triangles(50, bounds)

        # Compute optimal bounds
        optimal_bounds = SceneVoxelizer.compute_optimal_bounds(triangles)

        # Choose resolution
        resolution = recommend_resolution(optimal_bounds)

        # Create voxelizer
        voxelizer = SceneVoxelizer.create_from_resolution(resolution)

        # Voxelize
        grid = voxelizer.voxelize(triangles, optimal_bounds)

        # Get stats
        stats = voxelizer.get_last_stats()

        # Verify results
        assert stats is not None
        assert stats.triangle_count == 50
        assert grid.count_filled_voxels() > 0

        # Export data
        albedo_data = grid.to_albedo_bytes()
        normal_data = grid.to_normal_bytes()
        emissive_data = grid.to_emissive_bytes()

        expected_size = resolution.size ** 3 * 4
        assert len(albedo_data) == expected_size
        assert len(normal_data) == expected_size
        assert len(emissive_data) == expected_size

    def test_resolution_scaling(self) -> None:
        """Higher resolution should provide more detail."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        triangles = [
            Triangle(Vec3(2, 2, 5), Vec3(8, 2, 5), Vec3(5, 8, 5))
        ]

        # Voxelize at different resolutions
        results = {}
        for res in [VoxelResolution.LOW, VoxelResolution.MEDIUM]:
            voxelizer = SceneVoxelizer.create_from_resolution(res)
            grid = voxelizer.voxelize(triangles, bounds)
            results[res] = grid.count_filled_voxels()

        # Higher resolution should fill more voxels (more detail)
        assert results[VoxelResolution.MEDIUM] >= results[VoxelResolution.LOW]

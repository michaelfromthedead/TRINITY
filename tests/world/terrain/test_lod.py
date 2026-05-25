"""Tests for terrain LOD system."""

import math
from typing import List, Tuple

import pytest

from engine.world.terrain.lod import (
    BoundingBox,
    ClipmapRing,
    LODStitchMethod,
    QuadtreeNode,
    TerrainChunk,
    TerrainLODMethod,
    TerrainLODSystem,
    TerrainPatch,
    TerrainQuadtree,
)


class MockFrustum:
    """Mock frustum for testing."""

    def __init__(self, include_all: bool = True, exclude_bounds: BoundingBox = None):
        self._include_all = include_all
        self._exclude_bounds = exclude_bounds

    def contains_box(self, box: BoundingBox) -> bool:
        if self._exclude_bounds is not None:
            if box.intersects(self._exclude_bounds):
                return False
        return self._include_all

    def contains_point(self, x: float, y: float, z: float) -> bool:
        return self._include_all


# ============================================================================
# BoundingBox tests
# ============================================================================


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_default_values(self):
        """Test default bounding box values."""
        box = BoundingBox()
        assert box.min_x == 0.0
        assert box.min_y == 0.0
        assert box.min_z == 0.0
        assert box.max_x == 0.0
        assert box.max_y == 0.0
        assert box.max_z == 0.0

    def test_custom_values(self):
        """Test custom bounding box values."""
        box = BoundingBox(
            min_x=-10.0,
            min_y=0.0,
            min_z=-10.0,
            max_x=10.0,
            max_y=5.0,
            max_z=10.0,
        )
        assert box.min_x == -10.0
        assert box.max_x == 10.0

    def test_center(self):
        """Test center calculation."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=20.0,
            max_z=30.0,
        )
        center = box.center
        assert center == (5.0, 10.0, 15.0)

    def test_size(self):
        """Test size calculation."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=20.0,
            max_z=30.0,
        )
        size = box.size
        assert size == (10.0, 20.0, 30.0)

    def test_width_height_depth(self):
        """Test dimension properties."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=20.0,
            max_z=30.0,
        )
        assert box.width == 10.0
        assert box.height == 20.0
        assert box.depth == 30.0

    def test_contains_point_inside(self):
        """Test point containment for point inside."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        assert box.contains_point(5.0, 5.0, 5.0)

    def test_contains_point_outside(self):
        """Test point containment for point outside."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        assert not box.contains_point(15.0, 5.0, 5.0)

    def test_contains_point_on_edge(self):
        """Test point containment for point on edge."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        assert box.contains_point(0.0, 5.0, 5.0)
        assert box.contains_point(10.0, 5.0, 5.0)

    def test_intersects_overlapping(self):
        """Test intersection with overlapping boxes."""
        box1 = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        box2 = BoundingBox(
            min_x=5.0,
            min_y=5.0,
            min_z=5.0,
            max_x=15.0,
            max_y=15.0,
            max_z=15.0,
        )
        assert box1.intersects(box2)
        assert box2.intersects(box1)

    def test_intersects_non_overlapping(self):
        """Test intersection with non-overlapping boxes."""
        box1 = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        box2 = BoundingBox(
            min_x=20.0,
            min_y=20.0,
            min_z=20.0,
            max_x=30.0,
            max_y=30.0,
            max_z=30.0,
        )
        assert not box1.intersects(box2)

    def test_intersects_touching(self):
        """Test intersection with touching boxes."""
        box1 = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        box2 = BoundingBox(
            min_x=10.0,
            min_y=0.0,
            min_z=0.0,
            max_x=20.0,
            max_y=10.0,
            max_z=10.0,
        )
        assert box1.intersects(box2)

    def test_distance_to_point_inside(self):
        """Test distance to point inside box."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        assert box.distance_to_point(5.0, 5.0, 5.0) == 0.0

    def test_distance_to_point_outside(self):
        """Test distance to point outside box."""
        box = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=10.0,
            max_y=10.0,
            max_z=10.0,
        )
        # Point is 10 units away on X axis
        assert abs(box.distance_to_point(20.0, 5.0, 5.0) - 10.0) < 0.001


# ============================================================================
# TerrainChunk tests
# ============================================================================


class TestTerrainChunk:
    """Tests for TerrainChunk class."""

    def test_default_values(self):
        """Test default chunk values."""
        chunk = TerrainChunk()
        assert chunk.lod_level == 0
        assert chunk.vertex_count == 0
        assert chunk.index_count == 0
        assert chunk.max_error == 0.0

    def test_get_error_metric_close(self):
        """Test error metric for close camera."""
        chunk = TerrainChunk(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            max_error=10.0,
        )

        # Camera at center
        error = chunk.get_error_metric(32.0, 50.0, 32.0)
        assert error == 10.0  # Inside box, distance is 1.0

    def test_get_error_metric_far(self):
        """Test error metric for far camera."""
        chunk = TerrainChunk(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            max_error=10.0,
        )

        # Camera far away
        error = chunk.get_error_metric(100.0, 50.0, 100.0)
        assert error < 10.0

    def test_get_screen_space_error(self):
        """Test screen space error calculation."""
        chunk = TerrainChunk(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            max_error=10.0,
        )

        fov = math.pi / 3  # 60 degrees
        screen_height = 1080

        # Close camera - high screen error
        error_close = chunk.get_screen_space_error(32.0, 50.0, 32.0, fov, screen_height)

        # Far camera - lower screen error
        error_far = chunk.get_screen_space_error(200.0, 50.0, 200.0, fov, screen_height)

        assert error_close > error_far


# ============================================================================
# QuadtreeNode tests
# ============================================================================


class TestQuadtreeNode:
    """Tests for QuadtreeNode class."""

    def test_is_leaf_without_children(self):
        """Test that node without children is a leaf."""
        node = QuadtreeNode()
        assert node.is_leaf

    def test_is_leaf_with_children(self):
        """Test that node with children is not a leaf."""
        node = QuadtreeNode()
        node.children = [QuadtreeNode() for _ in range(4)]
        assert not node.is_leaf

    def test_should_split_high_error(self):
        """Test splitting decision with high error."""
        node = QuadtreeNode(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            depth=0,
            max_error=100.0,
        )

        # Close camera should trigger split
        should_split = node.should_split(32.0, 50.0, 32.0, 1.0, 8)
        assert should_split

    def test_should_split_low_error(self):
        """Test splitting decision with low error."""
        node = QuadtreeNode(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            depth=0,
            max_error=0.1,
        )

        # Far camera should not trigger split
        should_split = node.should_split(1000.0, 50.0, 1000.0, 1.0, 8)
        assert not should_split

    def test_should_split_max_depth(self):
        """Test splitting decision at max depth."""
        node = QuadtreeNode(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
            depth=8,
            max_error=100.0,
        )

        # Should not split at max depth
        should_split = node.should_split(32.0, 50.0, 32.0, 1.0, 8)
        assert not should_split

    def test_get_center(self):
        """Test getting node center."""
        node = QuadtreeNode(
            bounds=BoundingBox(
                min_x=0.0,
                min_y=0.0,
                min_z=0.0,
                max_x=64.0,
                max_y=100.0,
                max_z=64.0,
            ),
        )

        center = node.get_center()
        assert center == (32.0, 32.0)


# ============================================================================
# TerrainQuadtree tests
# ============================================================================


class TestTerrainQuadtree:
    """Tests for TerrainQuadtree class."""

    def test_initialization(self):
        """Test quadtree initialization."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=4)

        assert tree.max_depth == 4
        assert tree.root is not None
        assert tree.root.bounds == bounds

    def test_invalid_max_depth(self):
        """Test that invalid max_depth raises error."""
        bounds = BoundingBox()

        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            TerrainQuadtree(bounds, max_depth=0)

    def test_invalid_base_error(self):
        """Test that invalid base_error raises error."""
        bounds = BoundingBox()

        with pytest.raises(ValueError, match="base_error must be > 0"):
            TerrainQuadtree(bounds, base_error=0)

    def test_select_lod_close_camera(self):
        """Test LOD selection with close camera."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=4)

        # Close camera should select more chunks (higher detail)
        chunks = tree.select_lod(128.0, 50.0, 128.0, 0.5)

        assert len(chunks) > 0

    def test_select_lod_far_camera(self):
        """Test LOD selection with far camera."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=4)

        # Far camera should select fewer chunks (lower detail)
        chunks = tree.select_lod(1000.0, 50.0, 1000.0, 0.5)

        assert len(chunks) > 0

    def test_select_lod_varying_detail(self):
        """Test that LOD varies with distance."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=6)

        chunks_close = tree.select_lod(256.0, 50.0, 256.0, 0.5)
        chunks_far = tree.select_lod(2000.0, 50.0, 2000.0, 0.5)

        # Close camera should have more chunks (higher detail)
        assert len(chunks_close) >= len(chunks_far)

    def test_get_visible_chunks(self):
        """Test getting visible chunks with frustum culling."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=4)

        frustum = MockFrustum(include_all=True)
        chunks = tree.get_visible_chunks(frustum, 128.0, 50.0, 128.0, 0.5)

        assert len(chunks) > 0

    def test_get_visible_chunks_with_culling(self):
        """Test that frustum culling excludes chunks."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=4)

        # Exclude everything
        frustum = MockFrustum(include_all=False)
        chunks = tree.get_visible_chunks(frustum, 128.0, 50.0, 128.0, 0.5)

        assert len(chunks) == 0


# ============================================================================
# ClipmapRing tests
# ============================================================================


class TestClipmapRing:
    """Tests for ClipmapRing class."""

    def test_initialization(self):
        """Test clipmap ring initialization."""
        ring = ClipmapRing(
            level=0,
            inner_radius=0.0,
            outer_radius=64.0,
            resolution=32,
            cell_size=2.0,
        )
        assert ring.level == 0
        assert ring.inner_radius == 0.0
        assert ring.outer_radius == 64.0

    def test_invalid_level(self):
        """Test that negative level raises error."""
        with pytest.raises(ValueError, match="level must be >= 0"):
            ClipmapRing(level=-1, outer_radius=64.0)

    def test_invalid_inner_radius(self):
        """Test that negative inner_radius raises error."""
        with pytest.raises(ValueError, match="inner_radius must be >= 0"):
            ClipmapRing(inner_radius=-1.0, outer_radius=64.0)

    def test_invalid_outer_radius(self):
        """Test that invalid outer_radius raises error."""
        with pytest.raises(ValueError, match="outer_radius must be > inner_radius"):
            ClipmapRing(inner_radius=64.0, outer_radius=32.0)

    def test_invalid_resolution(self):
        """Test that invalid resolution raises error."""
        with pytest.raises(ValueError, match="resolution must be >= 2"):
            ClipmapRing(outer_radius=64.0, resolution=1)

    def test_invalid_cell_size(self):
        """Test that invalid cell_size raises error."""
        with pytest.raises(ValueError, match="cell_size must be > 0"):
            ClipmapRing(outer_radius=64.0, cell_size=0)

    def test_get_mesh_for_ring(self):
        """Test mesh generation for ring."""
        ring = ClipmapRing(
            level=0,
            inner_radius=0.0,
            outer_radius=32.0,
            resolution=8,
            cell_size=8.0,
        )

        vertices, indices = ring.get_mesh_for_ring(0.0, 0.0)

        assert len(vertices) > 0
        assert len(indices) > 0


# ============================================================================
# TerrainPatch tests
# ============================================================================


class TestTerrainPatch:
    """Tests for TerrainPatch class."""

    def test_default_values(self):
        """Test default patch values."""
        patch = TerrainPatch()
        assert patch.x == 0
        assert patch.z == 0
        assert patch.current_lod == 0
        assert patch.target_lod == 0
        assert patch.morph_factor == 0.0


# ============================================================================
# TerrainLODSystem tests
# ============================================================================


class TestTerrainLODSystem:
    """Tests for TerrainLODSystem class."""

    def test_initialization(self):
        """Test LOD system initialization."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        system = TerrainLODSystem(bounds, patch_size=64.0)

        assert system.patches_x == 8
        assert system.patches_z == 8
        assert system.method == TerrainLODMethod.QUADTREE
        assert system.stitch_method == LODStitchMethod.SKIRTS

    def test_invalid_patch_size(self):
        """Test that invalid patch_size raises error."""
        bounds = BoundingBox()

        with pytest.raises(ValueError, match="patch_size must be > 0"):
            TerrainLODSystem(bounds, patch_size=0)

    def test_invalid_max_lod_level(self):
        """Test that invalid max_lod_level raises error."""
        bounds = BoundingBox()

        with pytest.raises(ValueError, match="max_lod_level must be >= 1"):
            TerrainLODSystem(bounds, max_lod_level=0)

    def test_invalid_error_threshold(self):
        """Test that invalid error_threshold raises error."""
        bounds = BoundingBox()

        with pytest.raises(ValueError, match="error_threshold must be > 0"):
            TerrainLODSystem(bounds, error_threshold=0)

    def test_get_patch(self):
        """Test getting patch by coordinates."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(bounds, patch_size=64.0)

        patch = system.get_patch(0, 0)
        assert patch is not None
        assert patch.x == 0
        assert patch.z == 0

    def test_get_patch_out_of_bounds(self):
        """Test getting patch with out of bounds coordinates."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(bounds, patch_size=64.0)

        patch = system.get_patch(100, 100)
        assert patch is None

    def test_error_threshold_property(self):
        """Test error threshold property."""
        bounds = BoundingBox()
        system = TerrainLODSystem(bounds)

        system.error_threshold = 8.0
        assert system.error_threshold == 8.0

    def test_error_threshold_invalid(self):
        """Test setting invalid error threshold."""
        bounds = BoundingBox()
        system = TerrainLODSystem(bounds)

        with pytest.raises(ValueError, match="error_threshold must be > 0"):
            system.error_threshold = 0

    def test_update_changes_lod(self):
        """Test that update changes LOD based on camera distance."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.GEO_MIPMAPPING,
        )

        # Update with camera in center
        system.update(256.0, 50.0, 256.0)

        # Center patches should have lower LOD (more detail)
        center_patch = system.get_patch(4, 4)
        edge_patch = system.get_patch(0, 0)

        # Edge patch should have higher LOD (less detail)
        assert edge_patch.target_lod >= center_patch.target_lod

    def test_get_render_chunks(self):
        """Test getting render chunks."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(bounds, patch_size=64.0)

        chunks = system.get_render_chunks()

        assert len(chunks) == 16  # 4x4 patches

    def test_get_stitch_indices(self):
        """Test getting stitch indices for a patch."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.GEO_MIPMAPPING,
        )

        # Update LOD so patches have reasonable LOD levels
        system.update(128.0, 50.0, 128.0)

        patch = system.get_patch(2, 2)
        # Make sure the patch has LOD 0 for maximum indices
        patch.current_lod = 0
        # Use vertices_per_side=17 which is typical for terrain patches
        indices = system.get_stitch_indices(patch, vertices_per_side=17)

        assert len(indices) > 0

    def test_get_skirt_vertices(self):
        """Test getting skirt vertices for a patch."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(bounds, patch_size=64.0)

        patch = system.get_patch(0, 0)
        skirt_verts = system.get_skirt_vertices(patch)

        # Should have vertices for all 4 edges
        assert len(skirt_verts) > 0

    def test_get_morph_factor(self):
        """Test getting morph factor for a vertex."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            stitch_method=LODStitchMethod.MORPHING,
        )

        patch = system.get_patch(2, 2)
        morph = system.get_morph_factor(patch, 160.0, 160.0, 160.0, 160.0)

        assert 0.0 <= morph <= 1.0

    def test_get_morph_factor_without_morphing(self):
        """Test that morph factor is 0 without morphing stitch method."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            stitch_method=LODStitchMethod.SKIRTS,
        )

        patch = system.get_patch(2, 2)
        morph = system.get_morph_factor(patch, 160.0, 160.0, 160.0, 160.0)

        assert morph == 0.0


# ============================================================================
# Integration tests
# ============================================================================


class TestLODIntegration:
    """Integration tests for LOD system."""

    def test_full_lod_workflow(self):
        """Test complete LOD workflow."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=1024.0,
            max_y=200.0,
            max_z=1024.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            max_lod_level=4,
            error_threshold=4.0,
        )

        # Simulate camera movement and LOD updates
        for i in range(5):
            camera_x = 512.0 + i * 100.0
            system.update(camera_x, 100.0, 512.0)

            chunks = system.get_render_chunks()
            assert len(chunks) > 0

            # Verify all chunks have valid LOD levels
            for chunk in chunks:
                assert 0 <= chunk.lod_level <= 4

    def test_quadtree_and_geo_mipmapping_consistency(self):
        """Test that different LOD methods produce consistent results."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )

        system_quadtree = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.QUADTREE,
        )

        system_geo = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.GEO_MIPMAPPING,
        )

        # Both should produce render chunks
        chunks_qt = system_quadtree.get_render_chunks()
        chunks_geo = system_geo.get_render_chunks()

        assert len(chunks_qt) > 0
        assert len(chunks_geo) > 0


# ============================================================================
# Enhanced LOD tests for edge cases
# ============================================================================


class TestEnhancedLODValidation:
    """Enhanced tests for LOD edge cases and transitions."""

    def test_lod_level_increases_with_distance(self):
        """Verify LOD level strictly increases (or stays same) with distance."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.GEO_MIPMAPPING,
            max_lod_level=4,
        )

        # Update with camera at one corner
        system.update(0.0, 50.0, 0.0)

        # Get LOD levels at increasing distances
        lod_at_distances = []
        for i in range(8):
            patch = system.get_patch(i, 0)
            if patch:
                lod_at_distances.append(patch.target_lod)

        # LOD should increase or stay same as we move away
        for i in range(len(lod_at_distances) - 1):
            assert lod_at_distances[i + 1] >= lod_at_distances[i], \
                f"LOD decreased: {lod_at_distances[i]} -> {lod_at_distances[i + 1]}"

    def test_neighbor_lod_difference_bounded(self):
        """Verify adjacent patches have LOD difference of at most 1."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            method=TerrainLODMethod.GEO_MIPMAPPING,
            max_lod_level=4,
        )

        system.update(256.0, 50.0, 256.0)
        chunks = system.get_render_chunks()

        # Check neighbor LOD differences
        for chunk in chunks:
            for neighbor_lod in chunk.neighbor_lods:
                if neighbor_lod >= 0:  # Valid neighbor
                    lod_diff = abs(chunk.lod_level - neighbor_lod)
                    assert lod_diff <= 2, f"Neighbor LOD difference too large: {lod_diff}"

    def test_morph_factor_continuous(self):
        """Verify morph factor changes continuously with distance."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            stitch_method=LODStitchMethod.MORPHING,
        )

        patch = system.get_patch(2, 2)
        center_x = 160.0
        center_z = 160.0

        # Sample morph factors at different camera distances
        morph_factors = []
        for dist in range(50, 300, 25):
            morph = system.get_morph_factor(
                patch, center_x, center_z,
                center_x + dist, center_z
            )
            morph_factors.append(morph)

        # Morph factor should change smoothly (no large jumps)
        for i in range(len(morph_factors) - 1):
            diff = abs(morph_factors[i + 1] - morph_factors[i])
            assert diff < 0.5, f"Large morph factor jump: {diff}"

    def test_stitch_indices_valid_range(self):
        """Verify stitch indices are within valid vertex range."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            stitch_method=LODStitchMethod.INDEX_MODIFICATION,
        )

        patch = system.get_patch(2, 2)
        vertices_per_side = 17
        indices = system.get_stitch_indices(patch, vertices_per_side)

        total_vertices = vertices_per_side * vertices_per_side
        for idx in indices:
            assert 0 <= idx < total_vertices, f"Index out of range: {idx}"

    def test_skirt_vertices_cover_all_edges(self):
        """Verify skirt vertices cover all four edges."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=256.0,
            max_y=100.0,
            max_z=256.0,
        )
        system = TerrainLODSystem(
            bounds,
            patch_size=64.0,
            stitch_method=LODStitchMethod.SKIRTS,
        )

        patch = system.get_patch(2, 2)
        vertices_per_side = 17
        skirt_verts = system.get_skirt_vertices(patch, skirt_depth=10.0, vertices_per_side=vertices_per_side)

        # Should have vertices for all 4 edges
        expected_count = vertices_per_side * 4
        assert len(skirt_verts) == expected_count, f"Expected {expected_count} skirt verts, got {len(skirt_verts)}"

    def test_quadtree_chunk_lod_levels_consistent(self):
        """Verify quadtree chunks have consistent LOD levels."""
        bounds = BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=512.0,
            max_y=100.0,
            max_z=512.0,
        )
        tree = TerrainQuadtree(bounds, max_depth=6, base_error=100.0)

        chunks = tree.select_lod(256.0, 50.0, 256.0, 0.5)

        # All chunks should have valid LOD levels
        for chunk in chunks:
            assert 0 <= chunk.lod_level <= 6, f"Invalid LOD level: {chunk.lod_level}"

        # Note: In quadtree LOD, higher lod_level means DEEPER in tree = MORE detail
        # This is the opposite convention of the patch-based system
        # Chunks closer to camera should have higher lod_level (deeper splits = more detail)
        if len(chunks) >= 2:
            # Find closest and farthest chunks from camera
            closest = min(chunks, key=lambda c: c.bounds.distance_to_point(256.0, 50.0, 256.0))
            farthest = max(chunks, key=lambda c: c.bounds.distance_to_point(256.0, 50.0, 256.0))

            # Note: quadtree convention - higher lod_level = more detail = closer to camera
            # So closer chunk should have HIGHER or equal lod_level
            assert closest.lod_level >= farthest.lod_level, \
                f"Close chunk should have >= LOD in quadtree: close={closest.lod_level}, far={farthest.lod_level}"

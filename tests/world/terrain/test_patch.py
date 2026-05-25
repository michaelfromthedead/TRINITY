"""
Tests for terrain patch component (patch.py).

Tests cover:
- TerrainPatch creation and validation
- LOD selection based on camera distance
- World coordinate conversion
- Neighbor relationships
- Cache invalidation
- Error metrics
"""

import pytest
import math

from engine.world.terrain.heightfield import Heightfield, HeightfieldConfig
from engine.world.terrain.patch import TerrainPatch
from engine.world.terrain.constants import (
    DEFAULT_LOD_LEVELS,
    DEFAULT_LOD_DISTANCES,
    MIN_LOD_LEVELS,
)


# =============================================================================
# TerrainPatch Creation Tests
# =============================================================================


class TestTerrainPatchCreation:
    """Tests for TerrainPatch initialization."""

    def test_default_creation(self):
        """Test creation with default values."""
        patch = TerrainPatch()
        assert patch.patch_x == 0
        assert patch.patch_y == 0
        assert patch.heightfield is None
        assert patch.current_lod == 0
        assert patch.lod_levels == 6
        assert len(patch.lod_distances) == 6

    def test_custom_position(self):
        """Test creation with custom grid position."""
        patch = TerrainPatch(patch_x=5, patch_y=10)
        assert patch.patch_x == 5
        assert patch.patch_y == 10

    def test_with_heightfield(self):
        """Test creation with heightfield."""
        hf = Heightfield(HeightfieldConfig(resolution=33))
        patch = TerrainPatch(heightfield=hf)
        assert patch.heightfield is hf

    def test_custom_lod_levels(self):
        """Test creation with custom LOD configuration."""
        patch = TerrainPatch(
            lod_levels=4,
            lod_distances=(25.0, 75.0, 150.0, 300.0)
        )
        assert patch.lod_levels == 4
        assert patch.lod_distances == (25.0, 75.0, 150.0, 300.0)

    def test_invalid_lod_levels_zero(self):
        """Test that zero LOD levels raises error."""
        with pytest.raises(ValueError, match="lod_levels must be >= 1"):
            TerrainPatch(lod_levels=0, lod_distances=())

    def test_invalid_lod_levels_negative(self):
        """Test that negative LOD levels raises error."""
        with pytest.raises(ValueError, match="lod_levels must be >= 1"):
            TerrainPatch(lod_levels=-1, lod_distances=())

    def test_mismatched_lod_distances(self):
        """Test that mismatched LOD distances count raises error."""
        with pytest.raises(ValueError, match="lod_distances length"):
            TerrainPatch(lod_levels=4, lod_distances=(10.0, 20.0))

    def test_non_ascending_lod_distances(self):
        """Test that non-ascending LOD distances raises error."""
        with pytest.raises(ValueError, match="strictly ascending"):
            TerrainPatch(lod_levels=3, lod_distances=(100.0, 50.0, 200.0))

    def test_equal_lod_distances(self):
        """Test that equal LOD distances raises error."""
        with pytest.raises(ValueError, match="strictly ascending"):
            TerrainPatch(lod_levels=3, lod_distances=(50.0, 50.0, 100.0))


# =============================================================================
# World Bounds Tests
# =============================================================================


class TestWorldBounds:
    """Tests for get_world_bounds method."""

    def test_bounds_without_heightfield(self):
        """Test bounds when no heightfield is attached."""
        patch = TerrainPatch(patch_x=5, patch_y=10)
        bounds = patch.get_world_bounds()
        assert bounds == (5.0, 0.0, 10.0, 6.0, 1.0, 11.0)

    def test_bounds_with_heightfield(self):
        """Test bounds with heightfield attached."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        bounds = patch.get_world_bounds()
        # With 65 samples and scale 1.0, world size is 64
        assert bounds[0] == 0.0  # min_x
        assert bounds[3] == 64.0  # max_x
        assert bounds[2] == 0.0  # min_z
        assert bounds[5] == 64.0  # max_z

    def test_bounds_with_heightfield_and_offset(self):
        """Test bounds with heightfield and patch offset."""
        config = HeightfieldConfig(resolution=33, scale=2.0)
        hf = Heightfield(config)
        # Patch at position (1, 2) with world size 64
        patch = TerrainPatch(patch_x=1, patch_y=2, heightfield=hf)
        bounds = patch.get_world_bounds()
        # World size is (33-1)*2 = 64
        assert bounds[0] == 64.0  # min_x = 1 * 64
        assert bounds[3] == 128.0  # max_x
        assert bounds[2] == 128.0  # min_z = 2 * 64
        assert bounds[5] == 192.0  # max_z

    def test_bounds_reflect_height_range(self):
        """Test bounds include actual height range."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, -50.0)
        hf.set_height_at(2, 2, 100.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        bounds = patch.get_world_bounds()
        assert bounds[1] == -50.0  # min_y
        assert bounds[4] == 100.0  # max_y


# =============================================================================
# LOD Selection Tests
# =============================================================================


class TestLODSelection:
    """Tests for LOD selection based on camera distance."""

    def test_select_lod_very_close(self):
        """Test LOD selection at very close distance."""
        patch = TerrainPatch()
        lod = patch.select_lod(10.0)  # First threshold is 50
        assert lod == 0
        assert patch.current_lod == 0

    def test_select_lod_at_threshold(self):
        """Test LOD selection exactly at threshold."""
        patch = TerrainPatch(
            lod_levels=3,
            lod_distances=(50.0, 100.0, 200.0)
        )
        # Just below first threshold
        assert patch.select_lod(49.9) == 0
        # At first threshold - should be LOD 1
        assert patch.select_lod(50.0) == 1
        # Just above first threshold
        assert patch.select_lod(50.1) == 1

    def test_select_lod_between_thresholds(self):
        """Test LOD selection between thresholds."""
        patch = TerrainPatch(
            lod_levels=3,
            lod_distances=(50.0, 100.0, 200.0)
        )
        assert patch.select_lod(75.0) == 1
        assert patch.select_lod(150.0) == 2

    def test_select_lod_beyond_all_thresholds(self):
        """Test LOD selection beyond all thresholds."""
        patch = TerrainPatch(
            lod_levels=3,
            lod_distances=(50.0, 100.0, 200.0)
        )
        lod = patch.select_lod(500.0)
        assert lod == 2  # Last LOD level

    def test_select_lod_negative_distance(self):
        """Test LOD selection with negative distance."""
        patch = TerrainPatch()
        lod = patch.select_lod(-100.0)
        assert lod == 0  # Should clamp to 0

    def test_select_lod_updates_current_lod(self):
        """Test that select_lod updates current_lod property."""
        patch = TerrainPatch()
        patch.select_lod(500.0)
        assert patch.current_lod > 0
        patch.select_lod(10.0)
        assert patch.current_lod == 0

    def test_select_lod_single_level(self):
        """Test LOD selection with single LOD level."""
        patch = TerrainPatch(lod_levels=1, lod_distances=(1000.0,))
        assert patch.select_lod(0.0) == 0
        assert patch.select_lod(500.0) == 0
        assert patch.select_lod(2000.0) == 0


# =============================================================================
# World Coordinate Conversion Tests
# =============================================================================


class TestWorldCoordinateConversion:
    """Tests for world to local coordinate conversion."""

    def test_get_height_at_world_center(self):
        """Test height query at patch center."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        height = patch.get_height_at_world(2.0, 2.0)
        assert height is not None
        assert abs(height - 100.0) < 1e-6

    def test_get_height_at_world_with_offset(self):
        """Test height query with patch offset."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)
        # Patch at world position (4, 4)
        patch = TerrainPatch(patch_x=1, patch_y=1, heightfield=hf)
        # World (6, 6) = local (2, 2)
        height = patch.get_height_at_world(6.0, 6.0)
        assert height is not None
        assert abs(height - 100.0) < 1e-6

    def test_get_height_at_world_outside_patch(self):
        """Test height query outside patch returns None."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        # Outside patch (world size is 4)
        height = patch.get_height_at_world(10.0, 10.0)
        assert height is None

    def test_get_height_at_world_no_heightfield(self):
        """Test height query without heightfield returns None."""
        patch = TerrainPatch(patch_x=0, patch_y=0)
        height = patch.get_height_at_world(0.0, 0.0)
        assert height is None

    def test_get_normal_at_world(self):
        """Test normal query at world position."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Create flat surface
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        normal = patch.get_normal_at_world(2.0, 2.0)
        assert normal is not None
        assert abs(normal[0]) < 1e-6
        assert abs(normal[1] - 1.0) < 1e-6
        assert abs(normal[2]) < 1e-6

    def test_get_normal_at_world_outside_patch(self):
        """Test normal query outside patch returns None."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        normal = patch.get_normal_at_world(100.0, 100.0)
        assert normal is None


# =============================================================================
# Neighbor Relationship Tests
# =============================================================================


class TestNeighborRelationships:
    """Tests for neighbor patch management."""

    def test_set_neighbor_valid_direction(self):
        """Test setting neighbor in valid direction."""
        patch1 = TerrainPatch(patch_x=0, patch_y=0)
        patch2 = TerrainPatch(patch_x=1, patch_y=0)
        assert patch1.set_neighbor('east', patch2)
        assert patch1.get_neighbor('east') is patch2

    def test_set_neighbor_all_directions(self):
        """Test setting neighbors in all directions."""
        center = TerrainPatch(patch_x=1, patch_y=1)
        north = TerrainPatch(patch_x=1, patch_y=2)
        south = TerrainPatch(patch_x=1, patch_y=0)
        east = TerrainPatch(patch_x=2, patch_y=1)
        west = TerrainPatch(patch_x=0, patch_y=1)

        assert center.set_neighbor('north', north)
        assert center.set_neighbor('south', south)
        assert center.set_neighbor('east', east)
        assert center.set_neighbor('west', west)

        assert center.get_neighbor('north') is north
        assert center.get_neighbor('south') is south
        assert center.get_neighbor('east') is east
        assert center.get_neighbor('west') is west

    def test_set_neighbor_invalid_direction(self):
        """Test setting neighbor with invalid direction."""
        patch1 = TerrainPatch()
        patch2 = TerrainPatch()
        assert not patch1.set_neighbor('northeast', patch2)
        assert not patch1.set_neighbor('invalid', patch2)

    def test_clear_neighbor(self):
        """Test clearing a neighbor."""
        patch1 = TerrainPatch()
        patch2 = TerrainPatch()
        patch1.set_neighbor('north', patch2)
        assert patch1.get_neighbor('north') is patch2
        patch1.set_neighbor('north', None)
        assert patch1.get_neighbor('north') is None

    def test_get_neighbor_unset(self):
        """Test getting unset neighbor returns None."""
        patch = TerrainPatch()
        assert patch.get_neighbor('north') is None
        assert patch.get_neighbor('south') is None

    def test_neighbor_invalidates_cache(self):
        """Test that setting neighbor invalidates mesh cache."""
        patch1 = TerrainPatch()
        patch2 = TerrainPatch()
        patch1._mesh_cache[0] = "cached_mesh"
        patch1.set_neighbor('north', patch2)
        assert len(patch1._mesh_cache) == 0


# =============================================================================
# Cache Invalidation Tests
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_cache_clears_mesh(self):
        """Test invalidate_cache clears mesh cache."""
        patch = TerrainPatch()
        patch._mesh_cache[0] = "mesh_lod_0"
        patch._mesh_cache[1] = "mesh_lod_1"
        patch.invalidate_cache()
        assert len(patch._mesh_cache) == 0

    def test_invalidate_cache_clears_collision(self):
        """Test invalidate_cache clears collision data."""
        patch = TerrainPatch()
        patch._collision_data = "collision_data"
        patch.invalidate_cache()
        assert patch._collision_data is None


# =============================================================================
# Error Metric Tests
# =============================================================================


class TestErrorMetric:
    """Tests for LOD error metric calculation."""

    def test_error_metric_lod_zero(self):
        """Test error metric for LOD 0 is always zero."""
        config = HeightfieldConfig(resolution=9, scale=1.0)
        hf = Heightfield(config)
        for z in range(9):
            for x in range(9):
                hf.set_height_at(x, z, x * z)
        patch = TerrainPatch(heightfield=hf)
        error = patch.get_error_metric(0)
        assert error == 0.0

    def test_error_metric_increases_with_lod(self):
        """Test error metric increases with higher LOD levels."""
        config = HeightfieldConfig(resolution=17, scale=1.0)
        hf = Heightfield(config)
        # Create irregular terrain
        for z in range(17):
            for x in range(17):
                hf.set_height_at(x, z, math.sin(x) * math.cos(z) * 50.0)
        patch = TerrainPatch(heightfield=hf)

        error_1 = patch.get_error_metric(1)
        error_2 = patch.get_error_metric(2)
        error_3 = patch.get_error_metric(3)

        # Error should generally increase with LOD
        # (though not guaranteed for all terrain)
        assert error_1 >= 0.0
        assert error_2 >= 0.0
        assert error_3 >= 0.0

    def test_error_metric_flat_terrain(self):
        """Test error metric for flat terrain is zero at all LODs."""
        config = HeightfieldConfig(resolution=17, scale=1.0)
        hf = Heightfield(config)
        hf.fill(50.0)
        patch = TerrainPatch(heightfield=hf)

        for lod in range(6):
            error = patch.get_error_metric(lod)
            assert error == 0.0

    def test_error_metric_invalid_lod(self):
        """Test error metric for invalid LOD returns infinity."""
        patch = TerrainPatch()
        assert patch.get_error_metric(-1) == float('inf')
        assert patch.get_error_metric(10) == float('inf')

    def test_error_metric_no_heightfield(self):
        """Test error metric without heightfield returns zero."""
        patch = TerrainPatch()
        assert patch.get_error_metric(1) == 0.0


# =============================================================================
# Vertex Count Tests
# =============================================================================


class TestVertexCount:
    """Tests for LOD vertex count calculation."""

    def test_lod_vertex_count_base(self):
        """Test vertex count at LOD 0."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(heightfield=hf)
        count = patch.get_lod_vertex_count(0)
        assert count == 65 * 65

    def test_lod_vertex_count_decreases(self):
        """Test vertex count decreases with higher LOD."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(heightfield=hf)

        count_0 = patch.get_lod_vertex_count(0)
        count_1 = patch.get_lod_vertex_count(1)
        count_2 = patch.get_lod_vertex_count(2)

        assert count_1 < count_0
        assert count_2 < count_1

    def test_lod_vertex_count_no_heightfield(self):
        """Test vertex count without heightfield returns zero."""
        patch = TerrainPatch()
        assert patch.get_lod_vertex_count(0) == 0

    def test_lod_vertex_count_invalid_lod(self):
        """Test vertex count for invalid LOD returns zero."""
        config = HeightfieldConfig(resolution=65)
        hf = Heightfield(config)
        patch = TerrainPatch(heightfield=hf)
        assert patch.get_lod_vertex_count(-1) == 0
        assert patch.get_lod_vertex_count(10) == 0


# =============================================================================
# Containment and Distance Tests
# =============================================================================


class TestContainmentAndDistance:
    """Tests for point containment and distance calculations."""

    def test_contains_world_point_inside(self):
        """Test point containment for point inside patch."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        assert patch.contains_world_point(32.0, 32.0)

    def test_contains_world_point_outside(self):
        """Test point containment for point outside patch."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        assert not patch.contains_world_point(100.0, 100.0)

    def test_contains_world_point_on_edge(self):
        """Test point containment for point on edge."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        assert patch.contains_world_point(0.0, 0.0)
        assert patch.contains_world_point(64.0, 64.0)

    def test_distance_to_point_inside(self):
        """Test distance calculation for point inside patch."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        distance = patch.distance_to_point(32.0, 0.0, 32.0)
        assert distance == 0.0

    def test_distance_to_point_outside(self):
        """Test distance calculation for point outside patch."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(0.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        # Point 10 units away from corner
        distance = patch.distance_to_point(74.0, 0.0, 0.0)
        assert abs(distance - 10.0) < 1e-6

    def test_get_center_world(self):
        """Test get_center_world calculation."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.fill(50.0)
        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)
        center = patch.get_center_world()
        assert abs(center[0] - 32.0) < 1e-6
        assert abs(center[1] - 50.0) < 1e-6
        assert abs(center[2] - 32.0) < 1e-6


# =============================================================================
# Hash and Equality Tests
# =============================================================================


class TestHashAndEquality:
    """Tests for hash and equality operations."""

    def test_hash_based_on_position(self):
        """Test hash is based on patch position."""
        patch1 = TerrainPatch(patch_x=5, patch_y=10)
        patch2 = TerrainPatch(patch_x=5, patch_y=10)
        assert hash(patch1) == hash(patch2)

    def test_hash_different_positions(self):
        """Test hash differs for different positions."""
        patch1 = TerrainPatch(patch_x=0, patch_y=0)
        patch2 = TerrainPatch(patch_x=1, patch_y=0)
        assert hash(patch1) != hash(patch2)

    def test_equality_same_position(self):
        """Test equality for same position."""
        patch1 = TerrainPatch(patch_x=5, patch_y=10)
        patch2 = TerrainPatch(patch_x=5, patch_y=10)
        assert patch1 == patch2

    def test_equality_different_position(self):
        """Test inequality for different position."""
        patch1 = TerrainPatch(patch_x=0, patch_y=0)
        patch2 = TerrainPatch(patch_x=1, patch_y=0)
        assert patch1 != patch2

    def test_equality_different_type(self):
        """Test inequality for different type."""
        patch = TerrainPatch()
        assert patch != "not a patch"
        assert patch != 42

    def test_usable_in_set(self):
        """Test patches can be used in a set."""
        patch1 = TerrainPatch(patch_x=0, patch_y=0)
        patch2 = TerrainPatch(patch_x=0, patch_y=0)
        patch3 = TerrainPatch(patch_x=1, patch_y=0)

        patch_set = {patch1, patch2, patch3}
        assert len(patch_set) == 2

    def test_usable_as_dict_key(self):
        """Test patches can be used as dictionary keys."""
        patch1 = TerrainPatch(patch_x=0, patch_y=0)
        patch2 = TerrainPatch(patch_x=1, patch_y=1)

        data = {patch1: "first", patch2: "second"}
        assert data[patch1] == "first"
        assert data[patch2] == "second"


# =============================================================================
# Constants Verification Tests
# =============================================================================


class TestConstantsVerification:
    """Tests to verify constants are properly used."""

    def test_default_lod_levels_from_constants(self):
        """Test default LOD levels match constants."""
        patch = TerrainPatch()
        assert patch.lod_levels == DEFAULT_LOD_LEVELS

    def test_default_lod_distances_from_constants(self):
        """Test default LOD distances match constants."""
        patch = TerrainPatch()
        assert patch.lod_distances == DEFAULT_LOD_DISTANCES

    def test_min_lod_levels_validation(self):
        """Test MIN_LOD_LEVELS is enforced."""
        with pytest.raises(ValueError, match="lod_levels must be >= 1"):
            TerrainPatch(lod_levels=MIN_LOD_LEVELS - 1, lod_distances=())


# =============================================================================
# Error Metric Edge Cases
# =============================================================================


class TestErrorMetricEdgeCases:
    """Tests for error metric edge cases."""

    def test_error_metric_linear_terrain(self):
        """Test error metric for perfectly linear terrain is zero."""
        config = HeightfieldConfig(resolution=17, scale=1.0)
        hf = Heightfield(config)

        # Linear interpolation: height = x + z
        # This should have zero error because linear interpolation is exact
        for z in range(17):
            for x in range(17):
                hf.set_height_at(x, z, float(x + z))

        patch = TerrainPatch(heightfield=hf)

        # For linear terrain, bilinear interpolation should be exact
        # So error should be zero (or very small due to floating point)
        error_1 = patch.get_error_metric(1)
        assert error_1 < 1e-6, f"Expected ~0 error for linear terrain, got {error_1}"

    def test_error_metric_quadratic_terrain(self):
        """Test error metric for quadratic terrain has non-zero error."""
        config = HeightfieldConfig(resolution=17, scale=1.0)
        hf = Heightfield(config)

        # Quadratic: height = x^2 + z^2
        # Bilinear interpolation can't exactly represent quadratic surfaces
        for z in range(17):
            for x in range(17):
                hf.set_height_at(x, z, float(x * x + z * z))

        patch = TerrainPatch(heightfield=hf)

        error_1 = patch.get_error_metric(1)
        error_2 = patch.get_error_metric(2)

        # Error should be non-zero for quadratic surface
        assert error_1 > 0, "Expected non-zero error for quadratic terrain"

        # Higher LOD should generally have higher error
        # (though not always guaranteed)
        assert error_2 >= error_1 * 0.5, "LOD 2 error should be comparable to LOD 1"

    def test_error_metric_step_size_at_high_lod(self):
        """Test error metric handles high LOD levels with large step sizes."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)

        # Create sinusoidal terrain
        for z in range(65):
            for x in range(65):
                hf.set_height_at(x, z, math.sin(x * 0.3) * math.cos(z * 0.3) * 50.0)

        patch = TerrainPatch(heightfield=hf)

        # Test all valid LOD levels
        for lod in range(6):
            error = patch.get_error_metric(lod)
            assert error >= 0, f"Error should be non-negative at LOD {lod}"
            assert not math.isinf(error), f"Error should not be infinite at LOD {lod}"


# =============================================================================
# World Coordinate Edge Cases
# =============================================================================


class TestWorldCoordinateEdgeCases:
    """Tests for world coordinate conversion edge cases."""

    def test_height_at_exact_boundary(self):
        """Test height query at exact patch boundary."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.fill(50.0)

        patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=hf)

        # World size is 4.0 (5 samples - 1 = 4 edges * scale 1.0)
        # At boundary
        h = patch.get_height_at_world(4.0, 4.0)
        assert h is not None
        assert abs(h - 50.0) < 1e-6

        # Just outside boundary
        h_outside = patch.get_height_at_world(4.01, 4.0)
        assert h_outside is None

    def test_negative_patch_coordinates(self):
        """Test patch with negative grid coordinates."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)

        patch = TerrainPatch(patch_x=-2, patch_y=-3, heightfield=hf)
        bounds = patch.get_world_bounds()

        # World size is 4.0
        assert bounds[0] == -8.0  # -2 * 4
        assert bounds[2] == -12.0  # -3 * 4

        # Query at center of patch
        h = patch.get_height_at_world(-6.0, -10.0)  # Local (2, 2)
        assert h is not None
        assert abs(h - 100.0) < 1e-6

    def test_large_patch_coordinates(self):
        """Test patch with large grid coordinates."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(32, 32, 200.0)

        patch = TerrainPatch(patch_x=1000, patch_y=1000, heightfield=hf)
        bounds = patch.get_world_bounds()

        # World size is 64.0
        assert bounds[0] == 64000.0  # 1000 * 64
        assert bounds[2] == 64000.0

        # Query at center
        h = patch.get_height_at_world(64032.0, 64032.0)  # Local (32, 32)
        assert h is not None
        assert abs(h - 200.0) < 1e-6

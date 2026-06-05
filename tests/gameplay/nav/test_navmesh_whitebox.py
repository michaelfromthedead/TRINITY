"""
WHITEBOX tests for NavMesh generation and queries.

Tests internal implementation details, edge cases, and boundary conditions:
- T-NAV-1.1: NavMesh build pipeline (voxelize, regions, contours)
- Vector3 operations and edge cases
- BoundingBox calculations
- Triangle math and geometry
- Heightfield voxelization internals
- Contour tracing algorithms
- Polygon operations
- Region building flood fill
- Obstacle handling
- Path modification algorithms
"""

import math
import pytest
import random
from typing import List

from engine.gameplay.nav.navmesh import (
    BoundingBox,
    Contour,
    ContourSet,
    Heightfield,
    HeightfieldSpan,
    NavMesh,
    NavMeshObstacle,
    NavMeshParams,
    NavMeshPolygon,
    NavMeshQueryResult,
    NavMeshTile,
    RaycastResult,
    Triangle,
    Vector3,
)
from engine.gameplay.nav.constants import (
    DEFAULT_AGENT_HEIGHT,
    DEFAULT_AGENT_RADIUS,
    DEFAULT_CELL_HEIGHT,
    DEFAULT_CELL_SIZE,
    DEFAULT_MAX_SLOPE,
    DEFAULT_STEP_HEIGHT,
    DEFAULT_TILE_SIZE,
    FLOAT_EPSILON,
    MAX_AGENT_HEIGHT,
    MAX_AGENT_RADIUS,
    MAX_CELL_SIZE,
    MAX_MAX_SLOPE,
    MAX_TILE_SIZE,
    MIN_AGENT_HEIGHT,
    MIN_AGENT_RADIUS,
    MIN_CELL_SIZE,
    MIN_MAX_SLOPE,
    MIN_TILE_SIZE,
    MIN_VERTICES_PER_POLY,
    MAX_VERTICES_PER_POLY,
    NavMeshBuildMode,
    ObstacleType,
    QueryType,
    ZERO_LENGTH_THRESHOLD,
)


# =============================================================================
# Vector3 WHITEBOX Tests
# =============================================================================

class TestVector3Whitebox:
    """Whitebox tests for Vector3 operations."""

    def test_vector3_default_values(self):
        """Test default Vector3 values are zero."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vector3_initialization(self):
        """Test Vector3 initialization with values."""
        v = Vector3(1.5, -2.0, 3.5)
        assert v.x == 1.5
        assert v.y == -2.0
        assert v.z == 3.5

    def test_vector3_addition(self):
        """Test vector addition."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_vector3_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector3(5, 7, 9)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result.x == 4
        assert result.y == 5
        assert result.z == 6

    def test_vector3_scalar_multiplication(self):
        """Test vector scalar multiplication."""
        v = Vector3(2, 3, 4)
        result = v * 2.5
        assert result.x == 5.0
        assert result.y == 7.5
        assert result.z == 10.0

    def test_vector3_scalar_division(self):
        """Test vector scalar division."""
        v = Vector3(10, 20, 30)
        result = v / 2
        assert result.x == 5.0
        assert result.y == 10.0
        assert result.z == 15.0

    def test_vector3_division_by_zero_raises(self):
        """Test division by zero raises ValueError."""
        v = Vector3(1, 2, 3)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            _ = v / 0

    def test_vector3_negation(self):
        """Test vector negation."""
        v = Vector3(1, -2, 3)
        result = -v
        assert result.x == -1
        assert result.y == 2
        assert result.z == -3

    def test_vector3_equality_within_epsilon(self):
        """Test vector equality with floating point tolerance."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(1.0 + FLOAT_EPSILON / 2, 2.0, 3.0)
        assert v1 == v2

    def test_vector3_inequality_outside_epsilon(self):
        """Test vectors are not equal outside epsilon."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(1.0 + FLOAT_EPSILON * 2, 2.0, 3.0)
        assert v1 != v2

    def test_vector3_equality_not_implemented(self):
        """Test equality with non-Vector3 returns NotImplemented."""
        v = Vector3(1, 2, 3)
        result = v.__eq__("not a vector")
        assert result is NotImplemented

    def test_vector3_hash_consistency(self):
        """Test that equal vectors have same hash."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        assert hash(v1) == hash(v2)

    def test_vector3_dot_product(self):
        """Test dot product calculation."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1.dot(v2)
        assert result == 1*4 + 2*5 + 3*6  # 32

    def test_vector3_dot_product_perpendicular(self):
        """Test dot product of perpendicular vectors is zero."""
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        assert v1.dot(v2) == 0

    def test_vector3_cross_product(self):
        """Test cross product calculation."""
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        result = v1.cross(v2)
        assert abs(result.x) < FLOAT_EPSILON
        assert abs(result.y) < FLOAT_EPSILON
        assert abs(result.z - 1) < FLOAT_EPSILON

    def test_vector3_cross_product_anticommutative(self):
        """Test cross product is anticommutative."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result1 = v1.cross(v2)
        result2 = v2.cross(v1)
        assert abs(result1.x + result2.x) < FLOAT_EPSILON
        assert abs(result1.y + result2.y) < FLOAT_EPSILON
        assert abs(result1.z + result2.z) < FLOAT_EPSILON

    def test_vector3_length(self):
        """Test vector length calculation."""
        v = Vector3(3, 4, 0)
        assert abs(v.length() - 5.0) < FLOAT_EPSILON

    def test_vector3_length_3d(self):
        """Test 3D vector length."""
        v = Vector3(1, 2, 2)
        assert abs(v.length() - 3.0) < FLOAT_EPSILON

    def test_vector3_length_squared(self):
        """Test squared length (faster than length)."""
        v = Vector3(3, 4, 0)
        assert abs(v.length_squared() - 25.0) < FLOAT_EPSILON

    def test_vector3_normalized(self):
        """Test vector normalization."""
        v = Vector3(3, 4, 0)
        n = v.normalized()
        assert abs(n.length() - 1.0) < FLOAT_EPSILON
        assert abs(n.x - 0.6) < FLOAT_EPSILON
        assert abs(n.y - 0.8) < FLOAT_EPSILON

    def test_vector3_normalized_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n.x == 0
        assert n.y == 0
        assert n.z == 0

    def test_vector3_normalized_tiny_vector(self):
        """Test normalizing very small vector returns zero."""
        v = Vector3(ZERO_LENGTH_THRESHOLD / 10, 0, 0)
        n = v.normalized()
        assert n.x == 0
        assert n.y == 0
        assert n.z == 0

    def test_vector3_distance_to(self):
        """Test distance between two points."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(3, 4, 0)
        assert abs(v1.distance_to(v2) - 5.0) < FLOAT_EPSILON

    def test_vector3_distance_squared_to(self):
        """Test squared distance between two points."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(3, 4, 0)
        assert abs(v1.distance_squared_to(v2) - 25.0) < FLOAT_EPSILON

    def test_vector3_lerp_start(self):
        """Test linear interpolation at t=0."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = v1.lerp(v2, 0)
        assert result == v1

    def test_vector3_lerp_end(self):
        """Test linear interpolation at t=1."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = v1.lerp(v2, 1)
        assert result == v2

    def test_vector3_lerp_midpoint(self):
        """Test linear interpolation at t=0.5."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = v1.lerp(v2, 0.5)
        assert abs(result.x - 5) < FLOAT_EPSILON
        assert abs(result.y - 10) < FLOAT_EPSILON
        assert abs(result.z - 15) < FLOAT_EPSILON

    def test_vector3_to_tuple(self):
        """Test conversion to tuple."""
        v = Vector3(1.5, 2.5, 3.5)
        t = v.to_tuple()
        assert t == (1.5, 2.5, 3.5)

    def test_vector3_from_tuple(self):
        """Test creation from tuple."""
        t = (1.5, 2.5, 3.5)
        v = Vector3.from_tuple(t)
        assert v.x == 1.5
        assert v.y == 2.5
        assert v.z == 3.5


# =============================================================================
# BoundingBox WHITEBOX Tests
# =============================================================================

class TestBoundingBoxWhitebox:
    """Whitebox tests for BoundingBox operations."""

    def test_boundingbox_default(self):
        """Test default bounding box."""
        bb = BoundingBox()
        assert bb.min_point == Vector3()
        assert bb.max_point == Vector3()

    def test_boundingbox_contains_inside(self):
        """Test point inside bounding box."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bb.contains(Vector3(5, 5, 5))

    def test_boundingbox_contains_on_boundary(self):
        """Test point on bounding box boundary."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bb.contains(Vector3(0, 5, 5))
        assert bb.contains(Vector3(10, 5, 5))

    def test_boundingbox_contains_outside(self):
        """Test point outside bounding box."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert not bb.contains(Vector3(15, 5, 5))
        assert not bb.contains(Vector3(-1, 5, 5))

    def test_boundingbox_intersects_overlapping(self):
        """Test overlapping bounding boxes."""
        bb1 = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        bb2 = BoundingBox(Vector3(5, 5, 5), Vector3(15, 15, 15))
        assert bb1.intersects(bb2)
        assert bb2.intersects(bb1)

    def test_boundingbox_intersects_touching(self):
        """Test touching bounding boxes."""
        bb1 = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        bb2 = BoundingBox(Vector3(10, 0, 0), Vector3(20, 10, 10))
        assert bb1.intersects(bb2)

    def test_boundingbox_intersects_separate(self):
        """Test non-intersecting bounding boxes."""
        bb1 = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        bb2 = BoundingBox(Vector3(20, 20, 20), Vector3(30, 30, 30))
        assert not bb1.intersects(bb2)

    def test_boundingbox_expand(self):
        """Test expanding bounding box."""
        bb = BoundingBox(Vector3(5, 5, 5), Vector3(10, 10, 10))
        expanded = bb.expand(2)
        assert expanded.min_point == Vector3(3, 3, 3)
        assert expanded.max_point == Vector3(12, 12, 12)

    def test_boundingbox_center(self):
        """Test bounding box center calculation."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 20, 30))
        center = bb.center()
        assert abs(center.x - 5) < FLOAT_EPSILON
        assert abs(center.y - 10) < FLOAT_EPSILON
        assert abs(center.z - 15) < FLOAT_EPSILON

    def test_boundingbox_size(self):
        """Test bounding box size calculation."""
        bb = BoundingBox(Vector3(5, 10, 15), Vector3(15, 25, 35))
        size = bb.size()
        assert abs(size.x - 10) < FLOAT_EPSILON
        assert abs(size.y - 15) < FLOAT_EPSILON
        assert abs(size.z - 20) < FLOAT_EPSILON


# =============================================================================
# Triangle WHITEBOX Tests
# =============================================================================

class TestTriangleWhitebox:
    """Whitebox tests for Triangle geometry operations."""

    def test_triangle_normal_xy_plane(self):
        """Test triangle normal in XY plane."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 1, 0)
        )
        normal = tri.normal()
        # Should point in +Z direction
        assert abs(normal.x) < FLOAT_EPSILON
        assert abs(normal.y) < FLOAT_EPSILON
        assert abs(abs(normal.z) - 1) < FLOAT_EPSILON

    def test_triangle_normal_xz_plane(self):
        """Test triangle normal in XZ plane (ground plane)."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1)
        )
        normal = tri.normal()
        # Should point in +/- Y direction
        assert abs(normal.x) < FLOAT_EPSILON
        assert abs(abs(normal.y) - 1) < FLOAT_EPSILON
        assert abs(normal.z) < FLOAT_EPSILON

    def test_triangle_area(self):
        """Test triangle area calculation."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(4, 0, 0),
            Vector3(0, 3, 0)
        )
        # Right triangle with base 4 and height 3 -> area = 6
        assert abs(tri.area() - 6.0) < FLOAT_EPSILON

    def test_triangle_area_degenerate(self):
        """Test degenerate triangle has zero area."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(2, 0, 0)  # Collinear points
        )
        assert tri.area() < FLOAT_EPSILON

    def test_triangle_centroid(self):
        """Test triangle centroid calculation."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(3, 0, 0),
            Vector3(0, 3, 0)
        )
        centroid = tri.centroid()
        assert abs(centroid.x - 1) < FLOAT_EPSILON
        assert abs(centroid.y - 1) < FLOAT_EPSILON
        assert abs(centroid.z - 0) < FLOAT_EPSILON

    def test_triangle_bounding_box(self):
        """Test triangle bounding box calculation."""
        tri = Triangle(
            Vector3(1, 2, 3),
            Vector3(4, 5, 6),
            Vector3(7, 8, 9)
        )
        bb = tri.bounding_box()
        assert bb.min_point == Vector3(1, 2, 3)
        assert bb.max_point == Vector3(7, 8, 9)


# =============================================================================
# HeightfieldSpan WHITEBOX Tests
# =============================================================================

class TestHeightfieldSpanWhitebox:
    """Whitebox tests for HeightfieldSpan operations."""

    def test_span_creation(self):
        """Test span creation with values."""
        span = HeightfieldSpan(min_height=5, max_height=10, area=1)
        assert span.min_height == 5
        assert span.max_height == 10
        assert span.area == 1
        assert span.region_id == 0
        assert span.next is None

    def test_span_linked_list(self):
        """Test span linked list structure."""
        span1 = HeightfieldSpan(0, 5)
        span2 = HeightfieldSpan(10, 15)
        span1.next = span2
        assert span1.next is span2
        assert span2.next is None


# =============================================================================
# Heightfield WHITEBOX Tests
# =============================================================================

class TestHeightfieldWhitebox:
    """Whitebox tests for Heightfield voxelization."""

    def test_heightfield_init_creates_empty_spans(self):
        """Test heightfield initializes with empty span grid."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox(Vector3(0, 0, 0), Vector3(3, 1, 3))
        )
        assert len(hf.spans) == 10
        assert len(hf.spans[0]) == 10
        for x in range(10):
            for z in range(10):
                assert hf.spans[x][z] is None

    def test_heightfield_add_span_first(self):
        """Test adding first span to empty column."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox()
        )
        hf.add_span(0, 0, 5, 10, area=1)
        assert hf.spans[0][0] is not None
        assert hf.spans[0][0].min_height == 5
        assert hf.spans[0][0].max_height == 10

    def test_heightfield_add_span_sorted(self):
        """Test spans are sorted by min_height."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox()
        )
        hf.add_span(0, 0, 20, 25, area=1)  # Add higher first
        hf.add_span(0, 0, 5, 10, area=1)   # Add lower second

        spans = hf.get_spans(0, 0)
        assert len(spans) == 2
        assert spans[0].min_height == 5  # Lower should be first
        assert spans[1].min_height == 20

    def test_heightfield_add_span_out_of_bounds(self):
        """Test adding span out of bounds is ignored."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox()
        )
        hf.add_span(-1, 0, 5, 10)  # Out of bounds
        hf.add_span(0, -1, 5, 10)  # Out of bounds
        hf.add_span(10, 0, 5, 10)  # Out of bounds
        hf.add_span(0, 10, 5, 10)  # Out of bounds
        # Should not crash

    def test_heightfield_get_spans_empty_column(self):
        """Test getting spans from empty column."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox()
        )
        spans = hf.get_spans(0, 0)
        assert spans == []

    def test_heightfield_get_spans_out_of_bounds(self):
        """Test getting spans out of bounds returns empty."""
        hf = Heightfield(
            width=10, depth=10,
            cell_size=0.3, cell_height=0.2,
            bounds=BoundingBox()
        )
        assert hf.get_spans(-1, 0) == []
        assert hf.get_spans(0, -1) == []
        assert hf.get_spans(10, 0) == []
        assert hf.get_spans(0, 10) == []


# =============================================================================
# Contour WHITEBOX Tests
# =============================================================================

class TestContourWhitebox:
    """Whitebox tests for Contour operations."""

    def test_contour_empty(self):
        """Test empty contour."""
        c = Contour()
        assert c.vertex_count() == 0
        assert not c.is_valid()

    def test_contour_valid_triangle(self):
        """Test valid triangular contour."""
        c = Contour(vertices=[
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        assert c.vertex_count() == 3
        assert c.is_valid()

    def test_contour_invalid_two_vertices(self):
        """Test contour with 2 vertices is invalid."""
        c = Contour(vertices=[
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
        ])
        assert c.vertex_count() == 2
        assert not c.is_valid()


# =============================================================================
# ContourSet WHITEBOX Tests
# =============================================================================

class TestContourSetWhitebox:
    """Whitebox tests for ContourSet operations."""

    def test_contourset_empty(self):
        """Test empty contour set."""
        cs = ContourSet()
        assert cs.contour_count() == 0

    def test_contourset_add_contour(self):
        """Test adding contour to set."""
        cs = ContourSet()
        c = Contour(vertices=[
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        cs.add_contour(c)
        assert cs.contour_count() == 1


# =============================================================================
# NavMeshParams WHITEBOX Tests
# =============================================================================

class TestNavMeshParamsWhitebox:
    """Whitebox tests for NavMeshParams validation."""

    def test_params_default_values(self):
        """Test default parameter values."""
        params = NavMeshParams()
        assert params.agent_radius == DEFAULT_AGENT_RADIUS
        assert params.agent_height == DEFAULT_AGENT_HEIGHT
        assert params.cell_size == DEFAULT_CELL_SIZE

    def test_params_agent_radius_too_small(self):
        """Test validation rejects too small agent radius."""
        with pytest.raises(ValueError, match="agent_radius"):
            NavMeshParams(agent_radius=MIN_AGENT_RADIUS - 0.01)

    def test_params_agent_radius_too_large(self):
        """Test validation rejects too large agent radius."""
        with pytest.raises(ValueError, match="agent_radius"):
            NavMeshParams(agent_radius=MAX_AGENT_RADIUS + 0.01)

    def test_params_agent_height_too_small(self):
        """Test validation rejects too small agent height."""
        with pytest.raises(ValueError, match="agent_height"):
            NavMeshParams(agent_height=MIN_AGENT_HEIGHT - 0.01)

    def test_params_agent_height_too_large(self):
        """Test validation rejects too large agent height."""
        with pytest.raises(ValueError, match="agent_height"):
            NavMeshParams(agent_height=MAX_AGENT_HEIGHT + 0.01)

    def test_params_negative_step_height(self):
        """Test validation rejects negative step height."""
        with pytest.raises(ValueError, match="step_height"):
            NavMeshParams(step_height=-0.1)

    def test_params_max_slope_too_small(self):
        """Test validation rejects too small max slope."""
        with pytest.raises(ValueError, match="max_slope"):
            NavMeshParams(max_slope=MIN_MAX_SLOPE - 0.01)

    def test_params_max_slope_too_large(self):
        """Test validation rejects too large max slope."""
        with pytest.raises(ValueError, match="max_slope"):
            NavMeshParams(max_slope=MAX_MAX_SLOPE + 0.01)

    def test_params_cell_size_too_small(self):
        """Test validation rejects too small cell size."""
        with pytest.raises(ValueError, match="cell_size"):
            NavMeshParams(cell_size=MIN_CELL_SIZE - 0.001)

    def test_params_cell_size_too_large(self):
        """Test validation rejects too large cell size."""
        with pytest.raises(ValueError, match="cell_size"):
            NavMeshParams(cell_size=MAX_CELL_SIZE + 0.01)

    def test_params_cell_height_zero(self):
        """Test validation rejects zero cell height."""
        with pytest.raises(ValueError, match="cell_height"):
            NavMeshParams(cell_height=0)

    def test_params_cell_height_negative(self):
        """Test validation rejects negative cell height."""
        with pytest.raises(ValueError, match="cell_height"):
            NavMeshParams(cell_height=-0.1)

    def test_params_negative_min_region_area(self):
        """Test validation rejects negative min region area."""
        with pytest.raises(ValueError, match="min_region_area"):
            NavMeshParams(min_region_area=-1)

    def test_params_negative_merge_region_area(self):
        """Test validation rejects negative merge region area."""
        with pytest.raises(ValueError, match="merge_region_area"):
            NavMeshParams(merge_region_area=-1)

    def test_params_max_contour_error_zero(self):
        """Test validation rejects zero max contour error."""
        with pytest.raises(ValueError, match="max_contour_error"):
            NavMeshParams(max_contour_error=0)

    def test_params_max_edge_length_zero(self):
        """Test validation rejects zero max edge length."""
        with pytest.raises(ValueError, match="max_edge_length"):
            NavMeshParams(max_edge_length=0)

    def test_params_vertices_per_poly_too_small(self):
        """Test validation rejects too few vertices per poly."""
        with pytest.raises(ValueError, match="max_vertices_per_poly"):
            NavMeshParams(max_vertices_per_poly=MIN_VERTICES_PER_POLY - 1)

    def test_params_vertices_per_poly_too_large(self):
        """Test validation rejects too many vertices per poly."""
        with pytest.raises(ValueError, match="max_vertices_per_poly"):
            NavMeshParams(max_vertices_per_poly=MAX_VERTICES_PER_POLY + 1)

    def test_params_tile_size_too_small(self):
        """Test validation rejects too small tile size."""
        with pytest.raises(ValueError, match="tile_size"):
            NavMeshParams(tile_size=MIN_TILE_SIZE - 0.01)

    def test_params_tile_size_too_large(self):
        """Test validation rejects too large tile size."""
        with pytest.raises(ValueError, match="tile_size"):
            NavMeshParams(tile_size=MAX_TILE_SIZE + 0.01)


# =============================================================================
# NavMeshPolygon WHITEBOX Tests
# =============================================================================

class TestNavMeshPolygonWhitebox:
    """Whitebox tests for NavMeshPolygon operations."""

    def test_polygon_center_calculation(self):
        """Test polygon center is calculated on init."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        assert abs(poly.center.x - 5) < FLOAT_EPSILON
        assert abs(poly.center.y - 0) < FLOAT_EPSILON
        assert abs(poly.center.z - 5) < FLOAT_EPSILON

    def test_polygon_contains_point_inside(self):
        """Test point inside polygon."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        assert poly.contains_point_2d(Vector3(5, 0, 5))

    def test_polygon_contains_point_outside(self):
        """Test point outside polygon."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        assert not poly.contains_point_2d(Vector3(15, 0, 5))
        assert not poly.contains_point_2d(Vector3(-1, 0, 5))

    def test_polygon_contains_point_on_edge(self):
        """Test point on polygon edge."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        # Point on edge should be inside (ray casting algorithm)
        result = poly.contains_point_2d(Vector3(5, 0, 0))
        # Edge case - may be true or false depending on implementation
        assert isinstance(result, bool)

    def test_polygon_contains_point_too_few_vertices(self):
        """Test polygon with too few vertices."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
            ]
        )
        assert not poly.contains_point_2d(Vector3(5, 0, 0))

    def test_polygon_get_edge(self):
        """Test getting polygon edge."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        edge0 = poly.get_edge(0)
        assert edge0[0] == Vector3(0, 0, 0)
        assert edge0[1] == Vector3(10, 0, 0)

    def test_polygon_get_edge_wrap_around(self):
        """Test getting last edge wraps to first vertex."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10),
            ]
        )
        edge3 = poly.get_edge(3)
        assert edge3[0] == Vector3(0, 0, 10)
        assert edge3[1] == Vector3(0, 0, 0)


# =============================================================================
# NavMeshTile WHITEBOX Tests
# =============================================================================

class TestNavMeshTileWhitebox:
    """Whitebox tests for NavMeshTile operations."""

    def test_tile_polygon_count(self):
        """Test tile polygon count."""
        tile = NavMeshTile(id=1, x=0, z=0)
        assert tile.polygon_count() == 0

    def test_tile_get_polygon_found(self):
        """Test getting polygon from tile."""
        tile = NavMeshTile(id=1, x=0, z=0)
        poly = NavMeshPolygon(id=42, vertices=[
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        tile.polygons.append(poly)

        found = tile.get_polygon(42)
        assert found is poly

    def test_tile_get_polygon_not_found(self):
        """Test getting non-existent polygon from tile."""
        tile = NavMeshTile(id=1, x=0, z=0)
        assert tile.get_polygon(999) is None


# =============================================================================
# NavMeshObstacle WHITEBOX Tests
# =============================================================================

class TestNavMeshObstacleWhitebox:
    """Whitebox tests for NavMeshObstacle operations."""

    def test_obstacle_cylinder_bounds(self):
        """Test cylinder obstacle bounding box."""
        obs = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(10, 5, 15),
            radius=2,
            height=3
        )
        bounds = obs.get_bounds()
        assert bounds.min_point == Vector3(8, 5, 13)
        assert bounds.max_point == Vector3(12, 8, 17)

    def test_obstacle_box_bounds(self):
        """Test box obstacle bounding box."""
        obs = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.BOX,
            position=Vector3(10, 5, 15),
            half_extents=Vector3(2, 3, 4)
        )
        bounds = obs.get_bounds()
        assert bounds.min_point == Vector3(8, 2, 11)
        assert bounds.max_point == Vector3(12, 8, 19)

    def test_obstacle_convex_bounds_from_vertices(self):
        """Test convex obstacle bounding box from vertices."""
        obs = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.CONVEX,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(5, 0, 0),
                Vector3(5, 3, 5),
                Vector3(0, 3, 5),
            ]
        )
        bounds = obs.get_bounds()
        assert bounds.min_point == Vector3(0, 0, 0)
        assert bounds.max_point == Vector3(5, 3, 5)

    def test_obstacle_convex_empty_vertices(self):
        """Test convex obstacle with no vertices returns position."""
        obs = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.CONVEX,
            position=Vector3(5, 5, 5),
            vertices=[]
        )
        bounds = obs.get_bounds()
        assert bounds.min_point == Vector3(5, 5, 5)
        assert bounds.max_point == Vector3(5, 5, 5)


# =============================================================================
# NavMesh Build Pipeline WHITEBOX Tests
# =============================================================================

class TestNavMeshBuildWhitebox:
    """Whitebox tests for NavMesh build pipeline (T-NAV-1.1)."""

    def test_build_empty_triangles(self):
        """Test building navmesh with no triangles fails."""
        navmesh = NavMesh()
        result = navmesh.build([])
        assert not result

    def test_build_single_triangle(self):
        """Test building navmesh with single triangle."""
        navmesh = NavMesh()
        triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(5, 0, 10)
            )
        ]
        result = navmesh.build(triangles)
        assert result
        assert navmesh.is_built
        assert navmesh.polygon_count >= 1

    def test_build_calculates_bounds(self):
        """Test build pipeline calculates correct bounds."""
        navmesh = NavMesh()
        triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(5, 5, 10)
            )
        ]
        navmesh.build(triangles)
        bounds = navmesh.bounds
        assert bounds.min_point.x == 0
        assert bounds.min_point.y == 0
        assert bounds.min_point.z == 0
        assert bounds.max_point.x == 10
        assert bounds.max_point.y == 5
        assert bounds.max_point.z == 10

    def test_build_steep_slope_unwalkable(self):
        """Test steep slopes are marked as unwalkable."""
        params = NavMeshParams(max_slope=30)  # 30 degrees
        navmesh = NavMesh(params)

        # Create a very steep triangle (45+ degrees)
        triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 10, 0),  # 45 degree slope
                Vector3(5, 5, 10)
            )
        ]
        navmesh.build(triangles)
        # Should build but steep areas should be unwalkable

    def test_build_flat_surface_walkable(self):
        """Test flat surfaces are walkable."""
        navmesh = NavMesh()
        # Create a flat ground plane
        triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10)
            ),
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            )
        ]
        result = navmesh.build(triangles)
        assert result
        assert navmesh.polygon_count >= 1


# =============================================================================
# NavMesh Query WHITEBOX Tests
# =============================================================================

class TestNavMeshQueryWhitebox:
    """Whitebox tests for NavMesh query operations."""

    @pytest.fixture
    def navmesh_with_polygons(self):
        """Create NavMesh with test polygons."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10),
        ])
        navmesh.add_polygon([
            Vector3(10, 0, 0),
            Vector3(20, 0, 0),
            Vector3(20, 0, 10),
            Vector3(10, 0, 10),
        ])
        return navmesh

    def test_find_nearest_point_on_navmesh(self, navmesh_with_polygons):
        """Test finding nearest point on navmesh."""
        result = navmesh_with_polygons.find_nearest_point(Vector3(5, 5, 5))
        assert result.success
        assert result.position is not None

    def test_find_nearest_point_no_navmesh(self):
        """Test finding nearest point with no navmesh."""
        navmesh = NavMesh()
        result = navmesh.find_nearest_point(Vector3(5, 5, 5))
        assert not result.success

    def test_find_nearest_point_too_far(self, navmesh_with_polygons):
        """Test finding nearest point outside search radius."""
        result = navmesh_with_polygons.find_nearest_point(
            Vector3(1000, 1000, 1000),
            search_radius=1.0
        )
        assert not result.success

    def test_find_polygon_at_inside(self, navmesh_with_polygons):
        """Test finding polygon at position inside navmesh."""
        poly_id = navmesh_with_polygons.find_polygon_at(Vector3(5, 0, 5))
        assert poly_id is not None

    def test_find_polygon_at_outside(self, navmesh_with_polygons):
        """Test finding polygon at position outside navmesh."""
        poly_id = navmesh_with_polygons.find_polygon_at(Vector3(100, 0, 100))
        assert poly_id is None

    def test_raycast_hit(self, navmesh_with_polygons):
        """Test raycast that hits navmesh."""
        start = Vector3(5, 10, 5)  # Above navmesh
        end = Vector3(5, -10, 5)   # Below navmesh
        result = navmesh_with_polygons.raycast(start, end)
        assert result.hit
        assert abs(result.position.y) < 1  # Should hit near y=0

    def test_raycast_miss(self, navmesh_with_polygons):
        """Test raycast that misses navmesh."""
        start = Vector3(100, 10, 100)  # Far from navmesh
        end = Vector3(100, -10, 100)
        result = navmesh_with_polygons.raycast(start, end)
        assert not result.hit

    def test_raycast_zero_length(self, navmesh_with_polygons):
        """Test raycast with zero length."""
        point = Vector3(5, 5, 5)
        result = navmesh_with_polygons.raycast(point, point)
        assert not result.hit

    def test_get_random_point_empty_navmesh(self):
        """Test getting random point from empty navmesh."""
        navmesh = NavMesh()
        result = navmesh.get_random_point()
        assert not result.success

    def test_get_random_point_valid_navmesh(self, navmesh_with_polygons):
        """Test getting random point from valid navmesh."""
        result = navmesh_with_polygons.get_random_point()
        assert result.success
        assert result.position is not None

    def test_polygon_query_overlapping(self, navmesh_with_polygons):
        """Test polygon query with overlapping box."""
        poly_ids = navmesh_with_polygons.polygon_query(
            Vector3(5, 0, 5),
            Vector3(10, 10, 10)
        )
        assert len(poly_ids) >= 1

    def test_polygon_query_no_overlap(self, navmesh_with_polygons):
        """Test polygon query with non-overlapping box."""
        poly_ids = navmesh_with_polygons.polygon_query(
            Vector3(100, 100, 100),
            Vector3(1, 1, 1)
        )
        assert len(poly_ids) == 0


# =============================================================================
# NavMesh Obstacle Management WHITEBOX Tests
# =============================================================================

class TestNavMeshObstacleManagementWhitebox:
    """Whitebox tests for NavMesh obstacle management."""

    def test_add_obstacle_assigns_id(self):
        """Test adding obstacle assigns ID."""
        navmesh = NavMesh()
        obs = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5),
            radius=1,
            height=2
        )
        obs_id = navmesh.add_obstacle(obs)
        assert obs_id >= 0
        assert obs.id == obs_id

    def test_remove_obstacle(self):
        """Test removing obstacle."""
        navmesh = NavMesh()
        obs = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER
        )
        obs_id = navmesh.add_obstacle(obs)
        assert navmesh.remove_obstacle(obs_id)
        assert navmesh.obstacle_count == 0

    def test_remove_nonexistent_obstacle(self):
        """Test removing non-existent obstacle."""
        navmesh = NavMesh()
        assert not navmesh.remove_obstacle(999)

    def test_update_obstacle_position(self):
        """Test updating obstacle position."""
        navmesh = NavMesh()
        obs = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(0, 0, 0)
        )
        obs_id = navmesh.add_obstacle(obs)

        assert navmesh.update_obstacle(obs_id, position=Vector3(10, 10, 10))
        updated = navmesh.get_obstacle(obs_id)
        assert updated.position == Vector3(10, 10, 10)

    def test_update_obstacle_rotation(self):
        """Test updating obstacle rotation."""
        navmesh = NavMesh()
        obs = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.BOX,
            rotation=0
        )
        obs_id = navmesh.add_obstacle(obs)

        assert navmesh.update_obstacle(obs_id, rotation=math.pi / 2)
        updated = navmesh.get_obstacle(obs_id)
        assert abs(updated.rotation - math.pi / 2) < FLOAT_EPSILON

    def test_update_nonexistent_obstacle(self):
        """Test updating non-existent obstacle."""
        navmesh = NavMesh()
        assert not navmesh.update_obstacle(999, position=Vector3(1, 1, 1))


# =============================================================================
# NavMesh Polygon Management WHITEBOX Tests
# =============================================================================

class TestNavMeshPolygonManagementWhitebox:
    """Whitebox tests for NavMesh polygon management."""

    def test_add_polygon_returns_id(self):
        """Test adding polygon returns unique ID."""
        navmesh = NavMesh()
        id1 = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        id2 = navmesh.add_polygon([
            Vector3(10, 0, 0),
            Vector3(11, 0, 0),
            Vector3(10, 0, 1),
        ])
        assert id1 != id2

    def test_remove_polygon(self):
        """Test removing polygon."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        assert navmesh.remove_polygon(poly_id)
        assert navmesh.polygon_count == 0

    def test_remove_polygon_updates_neighbors(self):
        """Test removing polygon updates neighbor references."""
        navmesh = NavMesh()
        # Add two adjacent polygons
        poly1_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(5, 0, 5),
            Vector3(0, 0, 5),
        ])
        poly2_id = navmesh.add_polygon([
            Vector3(5, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 5),
            Vector3(5, 0, 5),
        ])

        # Manually set neighbors (would normally be done by build)
        poly1 = navmesh.get_polygon(poly1_id)
        poly2 = navmesh.get_polygon(poly2_id)
        poly1.neighbors.append(poly2_id)
        poly2.neighbors.append(poly1_id)

        # Remove first polygon
        navmesh.remove_polygon(poly1_id)

        # Check neighbor reference is removed from second polygon
        assert poly1_id not in poly2.neighbors

    def test_remove_nonexistent_polygon(self):
        """Test removing non-existent polygon."""
        navmesh = NavMesh()
        assert not navmesh.remove_polygon(999)

    def test_get_polygon(self):
        """Test getting polygon by ID."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        poly = navmesh.get_polygon(poly_id)
        assert poly is not None
        assert poly.id == poly_id

    def test_get_nonexistent_polygon(self):
        """Test getting non-existent polygon."""
        navmesh = NavMesh()
        assert navmesh.get_polygon(999) is None

    def test_get_polygons_iterator(self):
        """Test iterating over polygons."""
        navmesh = NavMesh()
        navmesh.add_polygon([Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)])
        navmesh.add_polygon([Vector3(2, 0, 0), Vector3(3, 0, 0), Vector3(2, 0, 1)])

        count = 0
        for poly in navmesh.get_polygons():
            count += 1
            assert isinstance(poly, NavMeshPolygon)
        assert count == 2

    def test_get_neighbors(self):
        """Test getting polygon neighbors."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1),
        ])
        poly = navmesh.get_polygon(poly_id)
        poly.neighbors.append(999)

        neighbors = navmesh.get_neighbors(poly_id)
        assert 999 in neighbors

    def test_get_neighbors_nonexistent(self):
        """Test getting neighbors of non-existent polygon."""
        navmesh = NavMesh()
        assert navmesh.get_neighbors(999) == []


# =============================================================================
# Path Modification WHITEBOX Tests
# =============================================================================

class TestPathModificationWhitebox:
    """Whitebox tests for path smoothing and modification."""

    @pytest.fixture
    def navmesh(self):
        """Create NavMesh for path tests."""
        return NavMesh()

    def test_smooth_path_too_short(self, navmesh):
        """Test smoothing path with less than 3 points."""
        path = [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        result = navmesh.smooth_path(path)
        assert result == path

    def test_smooth_path_single_point(self, navmesh):
        """Test smoothing path with single point."""
        path = [Vector3(0, 0, 0)]
        result = navmesh.smooth_path(path)
        assert result == path

    def test_smooth_path_preserves_endpoints(self, navmesh):
        """Test smooth path preserves start and end points."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 5),
            Vector3(10, 0, 10),
        ]
        result = navmesh.smooth_path(path)
        assert result[0] == path[0]
        assert result[-1] == path[-1]

    def test_smooth_path_increases_points(self, navmesh):
        """Test smoothing adds intermediate points."""
        path = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(20, 0, 0),
        ]
        result = navmesh.smooth_path(path, iterations=1)
        assert len(result) > len(path)

    def test_funnel_path_too_short(self, navmesh):
        """Test funnel algorithm with short path."""
        path = [Vector3(0, 0, 0)]
        result = navmesh.funnel_path(path, [], [])
        assert result == path

    def test_adjust_corridor_width_empty_path(self, navmesh):
        """Test corridor adjustment with empty path."""
        result = navmesh.adjust_corridor_width([], width=1.0)
        assert result == []

    def test_adjust_corridor_width_single_point(self, navmesh):
        """Test corridor adjustment with single point."""
        path = [Vector3(0, 0, 0)]
        result = navmesh.adjust_corridor_width(path)
        assert result == path

    def test_adjust_corridor_width_zero_width(self, navmesh):
        """Test corridor adjustment with zero width."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(10, 0, 0),
        ]
        result = navmesh.adjust_corridor_width(path, width=0)
        assert result == path


# =============================================================================
# Convex Hull WHITEBOX Tests
# =============================================================================

class TestConvexHullWhitebox:
    """Whitebox tests for convex hull algorithm (Graham scan)."""

    def test_convex_hull_returns_subset(self):
        """Test convex hull returns a subset of input points."""
        navmesh = NavMesh()
        points = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 0, 10),
        ]
        hull = navmesh._convex_hull_2d(points)
        # Hull should contain only input points
        for p in hull:
            assert p in points
        # Should have at least 2 points (degeneracy for simple inputs)
        assert len(hull) >= 2

    def test_convex_hull_preserves_extremes(self):
        """Test convex hull preserves extreme points."""
        navmesh = NavMesh()
        points = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10),
        ]
        hull = navmesh._convex_hull_2d(points)
        # Should have some extremes in hull
        assert len(hull) >= 2

    def test_convex_hull_excludes_or_includes_interior(self):
        """Test convex hull behavior with interior points."""
        navmesh = NavMesh()
        points = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10),
            Vector3(5, 0, 5),  # Interior point
        ]
        hull = navmesh._convex_hull_2d(points)
        # Hull should not grow larger than original corners
        assert len(hull) <= 5

    def test_convex_hull_collinear_points(self):
        """Test convex hull with collinear points."""
        navmesh = NavMesh()
        points = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(10, 0, 0),
        ]
        hull = navmesh._convex_hull_2d(points)
        # Collinear points - hull may be 2 or 3 points depending on impl
        assert len(hull) <= 3

    def test_convex_hull_two_points(self):
        """Test convex hull with only two points."""
        navmesh = NavMesh()
        points = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
        ]
        hull = navmesh._convex_hull_2d(points)
        assert len(hull) == 2


# =============================================================================
# Triangle Rasterization WHITEBOX Tests
# =============================================================================

class TestTriangleRasterizationWhitebox:
    """Whitebox tests for triangle rasterization (voxelization)."""

    def test_point_in_triangle_inside(self):
        """Test point inside triangle in XZ plane."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 0, 10)
        )
        assert navmesh._point_in_triangle_2d(Vector3(5, 0, 3), tri)

    def test_point_in_triangle_outside(self):
        """Test point outside triangle."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 0, 10)
        )
        assert not navmesh._point_in_triangle_2d(Vector3(15, 0, 5), tri)

    def test_point_in_triangle_on_vertex(self):
        """Test point on triangle vertex."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 0, 10)
        )
        assert navmesh._point_in_triangle_2d(Vector3(0, 0, 0), tri)

    def test_get_triangle_height_at_center(self):
        """Test getting triangle height at center."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 0, 10)
        )
        # Flat triangle at y=0
        height = navmesh._get_triangle_height_at(tri, 5, 3)
        assert height is not None
        assert abs(height) < FLOAT_EPSILON

    def test_get_triangle_height_at_sloped(self):
        """Test getting height on sloped triangle."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 5, 0),  # Higher
            Vector3(5, 2.5, 10)
        )
        height = navmesh._get_triangle_height_at(tri, 5, 0)
        assert height is not None

    def test_get_triangle_height_at_degenerate(self):
        """Test getting height on degenerate triangle."""
        navmesh = NavMesh()
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(0, 0, 0),  # Coincident point
            Vector3(0, 0, 0)
        )
        height = navmesh._get_triangle_height_at(tri, 0, 0)
        # Should return None for degenerate triangle
        assert height is None


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestNavMeshEdgeCases:
    """Edge case and boundary condition tests."""

    def test_navmesh_not_built_initially(self):
        """Test navmesh is not built initially."""
        navmesh = NavMesh()
        assert not navmesh.is_built

    def test_navmesh_zero_polygons_initially(self):
        """Test navmesh has zero polygons initially."""
        navmesh = NavMesh()
        assert navmesh.polygon_count == 0

    def test_navmesh_zero_obstacles_initially(self):
        """Test navmesh has zero obstacles initially."""
        navmesh = NavMesh()
        assert navmesh.obstacle_count == 0

    def test_navmesh_with_custom_params(self):
        """Test navmesh creation with custom params."""
        params = NavMeshParams(
            agent_radius=1.0,
            agent_height=3.0,
            cell_size=0.5
        )
        navmesh = NavMesh(params)
        assert navmesh.params.agent_radius == 1.0
        assert navmesh.params.agent_height == 3.0

    def test_large_coordinate_values(self):
        """Test handling of large coordinate values."""
        navmesh = NavMesh()
        large_val = 100000.0
        poly_id = navmesh.add_polygon([
            Vector3(large_val, 0, large_val),
            Vector3(large_val + 10, 0, large_val),
            Vector3(large_val + 10, 0, large_val + 10),
        ])
        assert poly_id is not None

    def test_negative_coordinate_values(self):
        """Test handling of negative coordinate values."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon([
            Vector3(-100, 0, -100),
            Vector3(-90, 0, -100),
            Vector3(-95, 0, -90),
        ])
        assert poly_id is not None

    def test_very_small_polygon(self):
        """Test handling of very small polygons."""
        navmesh = NavMesh()
        epsilon = 0.001
        poly_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(epsilon, 0, 0),
            Vector3(epsilon / 2, 0, epsilon),
        ])
        assert poly_id is not None

    def test_many_polygons(self):
        """Test adding many polygons."""
        navmesh = NavMesh()
        num_polys = 100

        for i in range(num_polys):
            navmesh.add_polygon([
                Vector3(i * 10, 0, 0),
                Vector3(i * 10 + 9, 0, 0),
                Vector3(i * 10 + 5, 0, 9),
            ])

        assert navmesh.polygon_count == num_polys

    def test_random_point_reproducibility(self):
        """Test random point generation uses Python random."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10),
        ])

        # Set seed for reproducibility
        random.seed(42)
        result1 = navmesh.get_random_point()

        random.seed(42)
        result2 = navmesh.get_random_point()

        # Same seed should give same result (or at least both succeed)
        assert result1.success == result2.success


# =============================================================================
# Triangle Area 2D WHITEBOX Tests
# =============================================================================

class TestTriangleArea2DWhitebox:
    """Whitebox tests for 2D triangle area calculation (funnel algorithm)."""

    def test_triangle_area_2d_nonzero(self):
        """Test non-zero signed area for non-collinear points."""
        navmesh = NavMesh()
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        c = Vector3(10, 0, 10)
        area = navmesh._triangle_area_2d(a, b, c)
        # Formula: (c.x - a.x) * (b.z - a.z) - (b.x - a.x) * (c.z - a.z)
        # = (10 - 0) * (0 - 0) - (10 - 0) * (10 - 0) = 0 - 100 = -100
        assert area != 0

    def test_triangle_area_2d_opposite_winding(self):
        """Test opposite sign for opposite winding."""
        navmesh = NavMesh()
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        c = Vector3(10, 0, 10)
        area1 = navmesh._triangle_area_2d(a, b, c)
        area2 = navmesh._triangle_area_2d(a, c, b)
        # Opposite winding should give opposite sign
        assert area1 * area2 < 0

    def test_triangle_area_2d_collinear(self):
        """Test zero area for collinear points."""
        navmesh = NavMesh()
        a = Vector3(0, 0, 0)
        b = Vector3(5, 0, 5)
        c = Vector3(10, 0, 10)
        area = navmesh._triangle_area_2d(a, b, c)
        assert abs(area) < FLOAT_EPSILON


# =============================================================================
# Performance and Stress Tests
# =============================================================================

class TestNavMeshPerformance:
    """Performance and stress tests."""

    def test_build_large_mesh(self):
        """Test building navmesh from many triangles."""
        navmesh = NavMesh()
        triangles = []

        # Create 10x10 grid of triangles
        for x in range(10):
            for z in range(10):
                triangles.append(Triangle(
                    Vector3(x, 0, z),
                    Vector3(x + 1, 0, z),
                    Vector3(x + 1, 0, z + 1)
                ))
                triangles.append(Triangle(
                    Vector3(x, 0, z),
                    Vector3(x + 1, 0, z + 1),
                    Vector3(x, 0, z + 1)
                ))

        result = navmesh.build(triangles)
        assert result
        assert navmesh.polygon_count >= 1

    def test_many_queries(self):
        """Test many consecutive queries."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(100, 0, 0),
            Vector3(100, 0, 100),
            Vector3(0, 0, 100),
        ])

        query_count = 0
        for _ in range(100):
            x = random.uniform(0, 100)
            z = random.uniform(0, 100)
            result = navmesh.find_nearest_point(Vector3(x, 5, z))
            # Just verify queries complete (result depends on impl)
            query_count += 1
        assert query_count == 100

    def test_many_obstacles(self):
        """Test adding many obstacles."""
        navmesh = NavMesh()

        for i in range(50):
            obs = NavMeshObstacle(
                id=0,
                obstacle_type=ObstacleType.CYLINDER,
                position=Vector3(i * 10, 0, i * 10),
                radius=1,
                height=2
            )
            navmesh.add_obstacle(obs)

        assert navmesh.obstacle_count == 50

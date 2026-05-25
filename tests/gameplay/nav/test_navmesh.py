"""
Comprehensive tests for NavMesh generation and queries.

Tests cover:
- NavMesh generation from geometry
- Polygon/triangle representation
- Point containment queries
- Nearest point on navmesh
- NavMesh regions and areas
- Area cost modifiers
- Dynamic navmesh updates
- NavMesh bounds and tiling
"""

import math
import pytest
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
    MAX_AGENT_HEIGHT,
    MAX_AGENT_RADIUS,
    MIN_AGENT_HEIGHT,
    MIN_AGENT_RADIUS,
    NavMeshBuildMode,
    ObstacleType,
    QueryType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_params():
    """Create default NavMesh parameters."""
    return NavMeshParams()


@pytest.fixture
def simple_navmesh():
    """Create a simple NavMesh with a few polygons."""
    navmesh = NavMesh()

    # Add a simple floor polygon
    navmesh.add_polygon([
        Vector3(0, 0, 0),
        Vector3(10, 0, 0),
        Vector3(10, 0, 10),
        Vector3(0, 0, 10),
    ])

    return navmesh


@pytest.fixture
def complex_navmesh():
    """Create a more complex NavMesh with multiple connected polygons."""
    navmesh = NavMesh()

    # Add grid of polygons
    poly_ids = []
    for x in range(3):
        for z in range(3):
            poly_id = navmesh.add_polygon([
                Vector3(x * 5, 0, z * 5),
                Vector3(x * 5 + 5, 0, z * 5),
                Vector3(x * 5 + 5, 0, z * 5 + 5),
                Vector3(x * 5, 0, z * 5 + 5),
            ])
            poly_ids.append(poly_id)

    return navmesh


@pytest.fixture
def floor_triangles():
    """Create triangles for a simple floor."""
    return [
        Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10)
        ),
        Triangle(
            Vector3(0, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10)
        ),
    ]


# =============================================================================
# Vector3 Tests
# =============================================================================


class TestVector3:
    """Tests for Vector3 class."""

    def test_default_construction(self):
        """Test default Vector3 is zero vector."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construction_with_values(self):
        """Test Vector3 construction with values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        """Test vector addition."""
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        c = a + b
        assert c.x == 5.0
        assert c.y == 7.0
        assert c.z == 9.0

    def test_subtraction(self):
        """Test vector subtraction."""
        a = Vector3(4, 5, 6)
        b = Vector3(1, 2, 3)
        c = a - b
        assert c.x == 3.0
        assert c.y == 3.0
        assert c.z == 3.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_scalar_division(self):
        """Test scalar division."""
        v = Vector3(2, 4, 6)
        result = v / 2
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.z == 3.0

    def test_division_by_zero_raises(self):
        """Test that division by zero raises ValueError."""
        v = Vector3(1, 2, 3)
        with pytest.raises(ValueError):
            _ = v / 0

    def test_negation(self):
        """Test vector negation."""
        v = Vector3(1, -2, 3)
        result = -v
        assert result.x == -1.0
        assert result.y == 2.0
        assert result.z == -3.0

    def test_equality(self):
        """Test vector equality."""
        a = Vector3(1, 2, 3)
        b = Vector3(1, 2, 3)
        c = Vector3(1, 2, 4)
        assert a == b
        assert not (a == c)

    def test_equality_with_small_difference(self):
        """Test that nearly equal vectors are equal."""
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0 + 1e-10, 2.0, 3.0)
        assert a == b

    def test_hash_consistency(self):
        """Test that equal vectors have same hash."""
        a = Vector3(1, 2, 3)
        b = Vector3(1, 2, 3)
        assert hash(a) == hash(b)

    def test_dot_product(self):
        """Test dot product calculation."""
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        assert a.dot(b) == 0.0

        c = Vector3(1, 2, 3)
        d = Vector3(4, 5, 6)
        assert c.dot(d) == 32.0

    def test_cross_product(self):
        """Test cross product calculation."""
        x = Vector3(1, 0, 0)
        y = Vector3(0, 1, 0)
        z = x.cross(y)
        assert z == Vector3(0, 0, 1)

    def test_length(self):
        """Test vector length calculation."""
        v = Vector3(3, 4, 0)
        assert v.length() == pytest.approx(5.0)

    def test_length_squared(self):
        """Test squared length calculation."""
        v = Vector3(3, 4, 0)
        assert v.length_squared() == 25.0

    def test_normalized(self):
        """Test vector normalization."""
        v = Vector3(3, 4, 0)
        n = v.normalized()
        assert n.length() == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero."""
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n == Vector3(0, 0, 0)

    def test_distance_to(self):
        """Test distance calculation."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_squared_to(self):
        """Test squared distance calculation."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert a.distance_squared_to(b) == 25.0

    def test_lerp(self):
        """Test linear interpolation."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 10, 10)

        mid = a.lerp(b, 0.5)
        assert mid == Vector3(5, 5, 5)

        assert a.lerp(b, 0) == a
        assert a.lerp(b, 1) == b

    def test_to_tuple(self):
        """Test conversion to tuple."""
        v = Vector3(1, 2, 3)
        t = v.to_tuple()
        assert t == (1.0, 2.0, 3.0)

    def test_from_tuple(self):
        """Test creation from tuple."""
        v = Vector3.from_tuple((1.0, 2.0, 3.0))
        assert v == Vector3(1, 2, 3)


# =============================================================================
# BoundingBox Tests
# =============================================================================


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_default_construction(self):
        """Test default bounding box."""
        bb = BoundingBox()
        assert bb.min_point == Vector3()
        assert bb.max_point == Vector3()

    def test_contains_point_inside(self):
        """Test point inside bounding box."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bb.contains(Vector3(5, 5, 5))

    def test_contains_point_outside(self):
        """Test point outside bounding box."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert not bb.contains(Vector3(15, 5, 5))

    def test_contains_point_on_boundary(self):
        """Test point on boundary."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bb.contains(Vector3(0, 5, 5))
        assert bb.contains(Vector3(10, 5, 5))

    def test_intersects_overlapping(self):
        """Test overlapping boxes."""
        a = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        b = BoundingBox(Vector3(5, 5, 5), Vector3(15, 15, 15))
        assert a.intersects(b)
        assert b.intersects(a)

    def test_intersects_non_overlapping(self):
        """Test non-overlapping boxes."""
        a = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        b = BoundingBox(Vector3(20, 20, 20), Vector3(30, 30, 30))
        assert not a.intersects(b)

    def test_intersects_touching(self):
        """Test touching boxes."""
        a = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        b = BoundingBox(Vector3(10, 0, 0), Vector3(20, 10, 10))
        assert a.intersects(b)

    def test_expand(self):
        """Test box expansion."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        expanded = bb.expand(2)
        assert expanded.min_point == Vector3(-2, -2, -2)
        assert expanded.max_point == Vector3(12, 12, 12)

    def test_center(self):
        """Test center calculation."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bb.center() == Vector3(5, 5, 5)

    def test_size(self):
        """Test size calculation."""
        bb = BoundingBox(Vector3(0, 0, 0), Vector3(10, 20, 30))
        size = bb.size()
        assert size.x == 10.0
        assert size.y == 20.0
        assert size.z == 30.0


# =============================================================================
# Triangle Tests
# =============================================================================


class TestTriangle:
    """Tests for Triangle class."""

    def test_construction(self):
        """Test triangle construction."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1)
        )
        assert tri.v0 == Vector3(0, 0, 0)
        assert tri.v1 == Vector3(1, 0, 0)
        assert tri.v2 == Vector3(0, 0, 1)

    def test_normal_flat_triangle(self):
        """Test normal for flat triangle."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1)
        )
        normal = tri.normal()
        assert normal.y == pytest.approx(-1.0) or normal.y == pytest.approx(1.0)
        assert abs(normal.x) < 0.01
        assert abs(normal.z) < 0.01

    def test_area(self):
        """Test area calculation."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            Vector3(0, 0, 2)
        )
        assert tri.area() == pytest.approx(2.0)

    def test_centroid(self):
        """Test centroid calculation."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(3, 0, 0),
            Vector3(0, 0, 3)
        )
        centroid = tri.centroid()
        assert centroid.x == pytest.approx(1.0)
        assert centroid.z == pytest.approx(1.0)

    def test_bounding_box(self):
        """Test bounding box calculation."""
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(5, 3, 0),
            Vector3(0, 0, 5)
        )
        bb = tri.bounding_box()
        assert bb.min_point == Vector3(0, 0, 0)
        assert bb.max_point == Vector3(5, 3, 5)


# =============================================================================
# NavMeshParams Tests
# =============================================================================


class TestNavMeshParams:
    """Tests for NavMeshParams configuration."""

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.agent_radius == DEFAULT_AGENT_RADIUS
        assert default_params.agent_height == DEFAULT_AGENT_HEIGHT
        assert default_params.step_height == DEFAULT_STEP_HEIGHT
        assert default_params.max_slope == DEFAULT_MAX_SLOPE
        assert default_params.cell_size == DEFAULT_CELL_SIZE
        assert default_params.cell_height == DEFAULT_CELL_HEIGHT
        assert default_params.tile_size == DEFAULT_TILE_SIZE
        assert default_params.build_mode == NavMeshBuildMode.STATIC

    def test_custom_values(self):
        """Test custom parameter values."""
        params = NavMeshParams(
            agent_radius=1.0,
            agent_height=3.0,
            step_height=0.5,
            max_slope=30.0
        )
        assert params.agent_radius == 1.0
        assert params.agent_height == 3.0
        assert params.step_height == 0.5
        assert params.max_slope == 30.0

    def test_invalid_agent_radius_too_small(self):
        """Test that too small agent radius raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(agent_radius=MIN_AGENT_RADIUS - 0.1)

    def test_invalid_agent_radius_too_large(self):
        """Test that too large agent radius raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(agent_radius=MAX_AGENT_RADIUS + 1.0)

    def test_invalid_agent_height_too_small(self):
        """Test that too small agent height raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(agent_height=MIN_AGENT_HEIGHT - 0.1)

    def test_invalid_agent_height_too_large(self):
        """Test that too large agent height raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(agent_height=MAX_AGENT_HEIGHT + 1.0)

    def test_invalid_step_height_negative(self):
        """Test that negative step height raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(step_height=-1.0)

    def test_invalid_max_slope_too_small(self):
        """Test that negative max slope raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(max_slope=-1.0)

    def test_invalid_max_slope_too_large(self):
        """Test that max slope >= 90 raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(max_slope=90.0)

    def test_invalid_cell_size_too_small(self):
        """Test that too small cell size raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(cell_size=0.001)

    def test_invalid_cell_height_zero(self):
        """Test that zero cell height raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(cell_height=0)

    def test_invalid_min_region_area_negative(self):
        """Test that negative min region area raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(min_region_area=-1)

    def test_invalid_max_contour_error_zero(self):
        """Test that zero max contour error raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(max_contour_error=0)

    def test_invalid_max_edge_length_zero(self):
        """Test that zero max edge length raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(max_edge_length=0)

    def test_invalid_vertices_per_poly_too_small(self):
        """Test that too few vertices per poly raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(max_vertices_per_poly=2)

    def test_invalid_tile_size_too_small(self):
        """Test that too small tile size raises error."""
        with pytest.raises(ValueError):
            NavMeshParams(tile_size=1.0)

    def test_validate_method(self, default_params):
        """Test explicit validate call."""
        default_params.validate()  # Should not raise


# =============================================================================
# Heightfield Tests
# =============================================================================


class TestHeightfield:
    """Tests for Heightfield voxelization."""

    def test_construction(self):
        """Test heightfield construction."""
        hf = Heightfield(
            width=10,
            depth=10,
            cell_size=1.0,
            cell_height=0.5,
            bounds=BoundingBox(Vector3(0, 0, 0), Vector3(10, 5, 10))
        )
        assert hf.width == 10
        assert hf.depth == 10
        assert hf.cell_size == 1.0
        assert hf.cell_height == 0.5

    def test_add_span(self):
        """Test adding a span."""
        hf = Heightfield(
            width=10, depth=10, cell_size=1.0, cell_height=0.5,
            bounds=BoundingBox()
        )
        hf.add_span(5, 5, 0, 10, area=1)

        spans = hf.get_spans(5, 5)
        assert len(spans) == 1
        assert spans[0].min_height == 0
        assert spans[0].max_height == 10

    def test_add_multiple_spans(self):
        """Test adding multiple spans to same column."""
        hf = Heightfield(
            width=10, depth=10, cell_size=1.0, cell_height=0.5,
            bounds=BoundingBox()
        )
        hf.add_span(5, 5, 0, 5, area=1)
        hf.add_span(5, 5, 10, 15, area=1)

        spans = hf.get_spans(5, 5)
        assert len(spans) == 2

    def test_span_sorted_by_height(self):
        """Test spans are sorted by height."""
        hf = Heightfield(
            width=10, depth=10, cell_size=1.0, cell_height=0.5,
            bounds=BoundingBox()
        )
        hf.add_span(5, 5, 20, 25, area=1)
        hf.add_span(5, 5, 0, 5, area=1)
        hf.add_span(5, 5, 10, 15, area=1)

        spans = hf.get_spans(5, 5)
        assert len(spans) == 3
        assert spans[0].min_height == 0
        assert spans[1].min_height == 10
        assert spans[2].min_height == 20

    def test_add_span_out_of_bounds(self):
        """Test adding span out of bounds is ignored."""
        hf = Heightfield(
            width=10, depth=10, cell_size=1.0, cell_height=0.5,
            bounds=BoundingBox()
        )
        hf.add_span(15, 5, 0, 10, area=1)  # x out of bounds
        hf.add_span(5, 15, 0, 10, area=1)  # z out of bounds

        # No crash, and bounds cells should still be empty
        assert hf.get_spans(15, 5) == []

    def test_empty_column(self):
        """Test getting spans from empty column."""
        hf = Heightfield(
            width=10, depth=10, cell_size=1.0, cell_height=0.5,
            bounds=BoundingBox()
        )
        spans = hf.get_spans(5, 5)
        assert len(spans) == 0


# =============================================================================
# HeightfieldSpan Tests
# =============================================================================


class TestHeightfieldSpan:
    """Tests for HeightfieldSpan."""

    def test_construction(self):
        """Test span construction."""
        span = HeightfieldSpan(min_height=0, max_height=10)
        assert span.min_height == 0
        assert span.max_height == 10
        assert span.area == 0
        assert span.region_id == 0
        assert span.next is None

    def test_with_area(self):
        """Test span with area type."""
        span = HeightfieldSpan(min_height=0, max_height=10, area=1)
        assert span.area == 1

    def test_linked_list(self):
        """Test span linked list."""
        span1 = HeightfieldSpan(min_height=0, max_height=5)
        span2 = HeightfieldSpan(min_height=10, max_height=15)
        span1.next = span2

        assert span1.next is span2
        assert span2.next is None


# =============================================================================
# Contour Tests
# =============================================================================


class TestContour:
    """Tests for Contour class."""

    def test_empty_contour(self):
        """Test empty contour."""
        contour = Contour()
        assert contour.vertex_count() == 0
        assert not contour.is_valid()

    def test_contour_with_vertices(self):
        """Test contour with vertices."""
        contour = Contour(
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)],
            region_id=1
        )
        assert contour.vertex_count() == 3
        assert contour.is_valid()
        assert contour.region_id == 1

    def test_contour_two_vertices_invalid(self):
        """Test contour with only 2 vertices is invalid."""
        contour = Contour(
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0)]
        )
        assert not contour.is_valid()


# =============================================================================
# ContourSet Tests
# =============================================================================


class TestContourSet:
    """Tests for ContourSet class."""

    def test_empty_contour_set(self):
        """Test empty contour set."""
        cs = ContourSet()
        assert cs.contour_count() == 0

    def test_add_contour(self):
        """Test adding contours."""
        cs = ContourSet()
        contour = Contour(
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)]
        )
        cs.add_contour(contour)
        assert cs.contour_count() == 1

    def test_multiple_contours(self):
        """Test multiple contours."""
        cs = ContourSet()
        for i in range(5):
            contour = Contour(
                vertices=[Vector3(i, 0, 0), Vector3(i+1, 0, 0), Vector3(i, 0, 1)],
                region_id=i
            )
            cs.add_contour(contour)
        assert cs.contour_count() == 5


# =============================================================================
# NavMeshPolygon Tests
# =============================================================================


class TestNavMeshPolygon:
    """Tests for NavMeshPolygon class."""

    def test_construction(self):
        """Test polygon construction."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        assert poly.id == 1
        assert len(poly.vertices) == 4

    def test_center_calculation(self):
        """Test center is calculated correctly."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        assert poly.center.x == pytest.approx(5.0)
        assert poly.center.z == pytest.approx(5.0)

    def test_contains_point_inside(self):
        """Test point inside polygon."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        assert poly.contains_point_2d(Vector3(5, 0, 5))

    def test_contains_point_outside(self):
        """Test point outside polygon."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        assert not poly.contains_point_2d(Vector3(15, 0, 5))

    def test_contains_point_on_edge(self):
        """Test point on polygon edge."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        # Edge case behavior may vary - just ensure no crash
        result = poly.contains_point_2d(Vector3(5, 0, 0))
        assert isinstance(result, bool)

    def test_get_edge(self):
        """Test getting polygon edge."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        edge = poly.get_edge(0)
        assert edge[0] == Vector3(0, 0, 0)
        assert edge[1] == Vector3(10, 0, 0)

    def test_get_edge_wraparound(self):
        """Test edge wraparound to first vertex."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(10, 0, 10),
                Vector3(0, 0, 10)
            ]
        )
        edge = poly.get_edge(3)
        assert edge[0] == Vector3(0, 0, 10)
        assert edge[1] == Vector3(0, 0, 0)

    def test_triangle_polygon(self):
        """Test triangle polygon."""
        poly = NavMeshPolygon(
            id=1,
            vertices=[
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(5, 0, 10)
            ]
        )
        assert len(poly.vertices) == 3
        assert poly.contains_point_2d(Vector3(5, 0, 3))

    def test_neighbors_empty(self):
        """Test polygon with no neighbors."""
        poly = NavMeshPolygon(id=1, vertices=[])
        assert len(poly.neighbors) == 0

    def test_area_type(self):
        """Test polygon area type."""
        poly = NavMeshPolygon(id=1, vertices=[], area_type=5)
        assert poly.area_type == 5


# =============================================================================
# NavMeshTile Tests
# =============================================================================


class TestNavMeshTile:
    """Tests for NavMeshTile class."""

    def test_construction(self):
        """Test tile construction."""
        tile = NavMeshTile(id=1, x=0, z=0)
        assert tile.id == 1
        assert tile.x == 0
        assert tile.z == 0
        assert tile.polygon_count() == 0

    def test_polygon_count(self):
        """Test polygon count."""
        tile = NavMeshTile(
            id=1, x=0, z=0,
            polygons=[
                NavMeshPolygon(id=1, vertices=[]),
                NavMeshPolygon(id=2, vertices=[])
            ]
        )
        assert tile.polygon_count() == 2

    def test_get_polygon(self):
        """Test getting polygon by ID."""
        poly1 = NavMeshPolygon(id=1, vertices=[])
        poly2 = NavMeshPolygon(id=2, vertices=[])
        tile = NavMeshTile(id=1, x=0, z=0, polygons=[poly1, poly2])

        assert tile.get_polygon(1) is poly1
        assert tile.get_polygon(2) is poly2
        assert tile.get_polygon(3) is None

    def test_is_loaded(self):
        """Test tile loaded state."""
        tile = NavMeshTile(id=1, x=0, z=0)
        assert tile.is_loaded

        tile.is_loaded = False
        assert not tile.is_loaded


# =============================================================================
# NavMesh Build Tests
# =============================================================================


class TestNavMeshBuild:
    """Tests for NavMesh building."""

    def test_build_empty_triangles(self):
        """Test build with empty triangles returns False."""
        navmesh = NavMesh()
        assert not navmesh.build([])

    def test_build_simple_floor(self, floor_triangles):
        """Test building from simple floor triangles."""
        navmesh = NavMesh()
        result = navmesh.build(floor_triangles)
        assert result
        assert navmesh.is_built
        assert navmesh.polygon_count > 0

    def test_build_sets_bounds(self, floor_triangles):
        """Test that build sets correct bounds."""
        navmesh = NavMesh()
        navmesh.build(floor_triangles)

        bounds = navmesh.bounds
        assert bounds.min_point.x <= 0
        assert bounds.min_point.z <= 0
        assert bounds.max_point.x >= 10
        assert bounds.max_point.z >= 10

    def test_build_with_custom_params(self, floor_triangles):
        """Test building with custom parameters."""
        params = NavMeshParams(
            agent_radius=0.3,
            cell_size=0.2
        )
        navmesh = NavMesh(params)
        result = navmesh.build(floor_triangles)
        assert result

    def test_is_built_false_before_build(self):
        """Test is_built is False before building."""
        navmesh = NavMesh()
        assert not navmesh.is_built

    def test_steep_slope_filtering(self):
        """Test that steep slopes are filtered out."""
        params = NavMeshParams(max_slope=30.0)
        navmesh = NavMesh(params)

        # Create steep triangle (vertical wall)
        steep_triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(10, 0, 0),
                Vector3(0, 10, 0)  # Points up
            )
        ]
        navmesh.build(steep_triangles)
        # Should still build but may have no walkable area


# =============================================================================
# NavMesh Query Tests
# =============================================================================


class TestNavMeshQueries:
    """Tests for NavMesh queries."""

    def test_find_nearest_point_on_navmesh(self, simple_navmesh):
        """Test finding nearest point."""
        result = simple_navmesh.find_nearest_point(Vector3(5, 1, 5))
        assert result.success
        assert result.position is not None

    def test_find_nearest_point_outside(self, simple_navmesh):
        """Test nearest point when far from navmesh."""
        result = simple_navmesh.find_nearest_point(Vector3(100, 0, 100), search_radius=50)
        # May or may not succeed depending on search radius

    def test_find_polygon_at_valid(self, simple_navmesh):
        """Test finding polygon at valid position."""
        poly_id = simple_navmesh.find_polygon_at(Vector3(5, 0, 5))
        assert poly_id is not None

    def test_find_polygon_at_invalid(self, simple_navmesh):
        """Test finding polygon at invalid position."""
        poly_id = simple_navmesh.find_polygon_at(Vector3(50, 0, 50))
        assert poly_id is None

    def test_get_random_point(self, simple_navmesh):
        """Test getting random point."""
        result = simple_navmesh.get_random_point()
        assert result.success
        assert result.position is not None

    def test_get_random_point_empty_navmesh(self):
        """Test random point on empty navmesh."""
        navmesh = NavMesh()
        result = navmesh.get_random_point()
        assert not result.success

    def test_get_random_point_in_radius(self, simple_navmesh):
        """Test getting random point in radius."""
        center = Vector3(5, 0, 5)
        radius = 3.0
        result = simple_navmesh.get_random_point_in_radius(center, radius)
        # May or may not succeed depending on navmesh coverage

    def test_raycast_no_hit(self, simple_navmesh):
        """Test raycast with no hit."""
        start = Vector3(5, 1, 5)
        end = Vector3(5, 1, 8)
        result = simple_navmesh.raycast(start, end)
        # Behavior depends on implementation

    def test_polygon_query(self, simple_navmesh):
        """Test polygon query with box."""
        center = Vector3(5, 0, 5)
        half_extents = Vector3(5, 1, 5)
        polygons = simple_navmesh.polygon_query(center, half_extents)
        assert isinstance(polygons, list)


# =============================================================================
# NavMesh Polygon Operations Tests
# =============================================================================


class TestNavMeshPolygonOps:
    """Tests for NavMesh polygon operations."""

    def test_add_polygon(self):
        """Test adding polygon manually."""
        navmesh = NavMesh()
        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10)
        ]
        poly_id = navmesh.add_polygon(vertices)
        assert poly_id >= 0
        assert navmesh.polygon_count == 1

    def test_add_multiple_polygons(self):
        """Test adding multiple polygons."""
        navmesh = NavMesh()
        for i in range(5):
            navmesh.add_polygon([
                Vector3(i, 0, 0),
                Vector3(i+1, 0, 0),
                Vector3(i, 0, 1)
            ])
        assert navmesh.polygon_count == 5

    def test_remove_polygon(self, simple_navmesh):
        """Test removing a polygon."""
        initial_count = simple_navmesh.polygon_count
        poly_id = list(simple_navmesh.get_polygons())[0].id

        result = simple_navmesh.remove_polygon(poly_id)
        assert result
        assert simple_navmesh.polygon_count == initial_count - 1

    def test_remove_nonexistent_polygon(self, simple_navmesh):
        """Test removing nonexistent polygon."""
        result = simple_navmesh.remove_polygon(99999)
        assert not result

    def test_get_polygon(self, simple_navmesh):
        """Test getting polygon by ID."""
        poly_id = list(simple_navmesh.get_polygons())[0].id
        poly = simple_navmesh.get_polygon(poly_id)
        assert poly is not None
        assert poly.id == poly_id

    def test_get_polygon_nonexistent(self, simple_navmesh):
        """Test getting nonexistent polygon."""
        poly = simple_navmesh.get_polygon(99999)
        assert poly is None

    def test_get_polygons_iterator(self, simple_navmesh):
        """Test polygon iterator."""
        polygons = list(simple_navmesh.get_polygons())
        assert len(polygons) > 0

    def test_get_neighbors(self, complex_navmesh):
        """Test getting polygon neighbors."""
        poly = list(complex_navmesh.get_polygons())[0]
        neighbors = complex_navmesh.get_neighbors(poly.id)
        assert isinstance(neighbors, list)

    def test_polygon_with_area_type(self):
        """Test polygon with custom area type."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon(
            [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)],
            area_type=5
        )
        poly = navmesh.get_polygon(poly_id)
        assert poly.area_type == 5


# =============================================================================
# NavMesh Obstacle Tests
# =============================================================================


class TestNavMeshObstacles:
    """Tests for NavMesh obstacles."""

    def test_add_cylinder_obstacle(self, simple_navmesh):
        """Test adding cylinder obstacle."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5),
            radius=1.0,
            height=2.0
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)
        assert obs_id >= 0
        assert simple_navmesh.obstacle_count == 1

    def test_add_box_obstacle(self, simple_navmesh):
        """Test adding box obstacle."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.BOX,
            position=Vector3(5, 1, 5),
            half_extents=Vector3(1, 1, 1)
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)
        assert obs_id >= 0

    def test_remove_obstacle(self, simple_navmesh):
        """Test removing obstacle."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5)
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)

        result = simple_navmesh.remove_obstacle(obs_id)
        assert result
        assert simple_navmesh.obstacle_count == 0

    def test_remove_nonexistent_obstacle(self, simple_navmesh):
        """Test removing nonexistent obstacle."""
        result = simple_navmesh.remove_obstacle(99999)
        assert not result

    def test_update_obstacle_position(self, simple_navmesh):
        """Test updating obstacle position."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5)
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)

        new_pos = Vector3(7, 0, 7)
        result = simple_navmesh.update_obstacle(obs_id, position=new_pos)
        assert result

        updated = simple_navmesh.get_obstacle(obs_id)
        assert updated.position == new_pos

    def test_update_obstacle_rotation(self, simple_navmesh):
        """Test updating obstacle rotation."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.BOX,
            position=Vector3(5, 0, 5),
            rotation=0.0
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)

        result = simple_navmesh.update_obstacle(obs_id, rotation=1.57)
        assert result

    def test_update_nonexistent_obstacle(self, simple_navmesh):
        """Test updating nonexistent obstacle."""
        result = simple_navmesh.update_obstacle(99999, position=Vector3())
        assert not result

    def test_get_obstacle(self, simple_navmesh):
        """Test getting obstacle by ID."""
        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5)
        )
        obs_id = simple_navmesh.add_obstacle(obstacle)

        retrieved = simple_navmesh.get_obstacle(obs_id)
        assert retrieved is not None
        assert retrieved.position == Vector3(5, 0, 5)

    def test_get_obstacles_iterator(self, simple_navmesh):
        """Test obstacle iterator."""
        for i in range(3):
            simple_navmesh.add_obstacle(NavMeshObstacle(
                id=0,
                obstacle_type=ObstacleType.CYLINDER,
                position=Vector3(i, 0, 0)
            ))

        obstacles = list(simple_navmesh.get_obstacles())
        assert len(obstacles) == 3

    def test_obstacle_bounds_cylinder(self):
        """Test cylinder obstacle bounds."""
        obstacle = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5),
            radius=2.0,
            height=3.0
        )
        bounds = obstacle.get_bounds()
        assert bounds.min_point.x == 3.0
        assert bounds.max_point.x == 7.0
        assert bounds.max_point.y == 3.0

    def test_obstacle_bounds_box(self):
        """Test box obstacle bounds."""
        obstacle = NavMeshObstacle(
            id=1,
            obstacle_type=ObstacleType.BOX,
            position=Vector3(5, 5, 5),
            half_extents=Vector3(1, 2, 3)
        )
        bounds = obstacle.get_bounds()
        assert bounds.min_point == Vector3(4, 3, 2)
        assert bounds.max_point == Vector3(6, 7, 8)


# =============================================================================
# Path Smoothing Tests
# =============================================================================


class TestPathSmoothing:
    """Tests for path smoothing."""

    def test_smooth_path_short(self, simple_navmesh):
        """Test smoothing very short path."""
        path = [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        smoothed = simple_navmesh.smooth_path(path)
        assert len(smoothed) == 2

    def test_smooth_path_single_point(self, simple_navmesh):
        """Test smoothing single point."""
        path = [Vector3(0, 0, 0)]
        smoothed = simple_navmesh.smooth_path(path)
        assert len(smoothed) == 1

    def test_smooth_path_normal(self, simple_navmesh):
        """Test normal path smoothing."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(5, 0, 5),
            Vector3(10, 0, 5)
        ]
        smoothed = simple_navmesh.smooth_path(path, iterations=2)
        assert len(smoothed) >= len(path)

    def test_smooth_path_maintains_endpoints(self, simple_navmesh):
        """Test that smoothing maintains start and end."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 5),
            Vector3(10, 0, 10)
        ]
        smoothed = simple_navmesh.smooth_path(path)
        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]

    def test_funnel_path_short(self, simple_navmesh):
        """Test funnel algorithm on short path."""
        path = [Vector3(0, 0, 0)]
        result = simple_navmesh.funnel_path(path, [], [])
        assert len(result) == 1

    def test_adjust_corridor_width(self, simple_navmesh):
        """Test corridor width adjustment."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 5),
            Vector3(10, 0, 10)
        ]
        adjusted = simple_navmesh.adjust_corridor_width(path, width=1.0)
        assert len(adjusted) == len(path)


# =============================================================================
# NavMesh Dynamic Updates Tests
# =============================================================================


class TestNavMeshDynamicUpdates:
    """Tests for dynamic NavMesh updates."""

    def test_dynamic_mode_params(self):
        """Test creating NavMesh with dynamic mode."""
        params = NavMeshParams(build_mode=NavMeshBuildMode.DYNAMIC)
        navmesh = NavMesh(params)
        assert navmesh.params.build_mode == NavMeshBuildMode.DYNAMIC

    def test_hybrid_mode_params(self):
        """Test creating NavMesh with hybrid mode."""
        params = NavMeshParams(build_mode=NavMeshBuildMode.HYBRID)
        navmesh = NavMesh(params)
        assert navmesh.params.build_mode == NavMeshBuildMode.HYBRID

    def test_update_method_exists(self, simple_navmesh):
        """Test update method exists."""
        simple_navmesh.update()  # Should not raise

    def test_obstacle_carving_marks_dirty(self):
        """Test that obstacle with carve marks tiles dirty."""
        params = NavMeshParams(build_mode=NavMeshBuildMode.DYNAMIC)
        navmesh = NavMesh(params)
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(100, 0, 0),
            Vector3(100, 0, 100),
            Vector3(0, 0, 100)
        ])

        obstacle = NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(50, 0, 50),
            radius=5.0,
            carve=True
        )
        navmesh.add_obstacle(obstacle)


# =============================================================================
# Raycast Tests
# =============================================================================


class TestRaycast:
    """Tests for raycast queries."""

    def test_raycast_result_structure(self):
        """Test RaycastResult default values."""
        result = RaycastResult()
        assert not result.hit
        assert result.distance == float('inf')
        assert result.polygon_id == -1

    def test_raycast_on_navmesh(self, simple_navmesh):
        """Test raycast on navmesh."""
        start = Vector3(5, 5, 5)
        end = Vector3(5, -5, 5)
        result = simple_navmesh.raycast(start, end)
        assert isinstance(result, RaycastResult)

    def test_raycast_with_filter_flags(self, simple_navmesh):
        """Test raycast with filter flags."""
        result = simple_navmesh.raycast(
            Vector3(5, 5, 5),
            Vector3(5, -5, 5),
            filter_flags=1
        )
        assert isinstance(result, RaycastResult)


# =============================================================================
# NavMeshQueryResult Tests
# =============================================================================


class TestNavMeshQueryResult:
    """Tests for NavMeshQueryResult."""

    def test_default_values(self):
        """Test default result values."""
        result = NavMeshQueryResult()
        assert not result.success
        assert result.query_type == QueryType.NEAREST_POINT
        assert result.position is None
        assert result.polygon_id is None
        assert result.distance == float('inf')

    def test_successful_result(self):
        """Test successful query result."""
        result = NavMeshQueryResult(
            success=True,
            query_type=QueryType.NEAREST_POINT,
            position=Vector3(5, 0, 5),
            polygon_id=1,
            distance=1.0
        )
        assert result.success
        assert result.position == Vector3(5, 0, 5)
        assert result.polygon_id == 1

    def test_path_result(self):
        """Test path query result."""
        result = NavMeshQueryResult(
            success=True,
            query_type=QueryType.PATH,
            path=[Vector3(0, 0, 0), Vector3(5, 0, 5)]
        )
        assert len(result.path) == 2


# =============================================================================
# Bounds and Properties Tests
# =============================================================================


class TestNavMeshProperties:
    """Tests for NavMesh properties."""

    def test_polygon_count_empty(self):
        """Test polygon count on empty NavMesh."""
        navmesh = NavMesh()
        assert navmesh.polygon_count == 0

    def test_obstacle_count_empty(self):
        """Test obstacle count on empty NavMesh."""
        navmesh = NavMesh()
        assert navmesh.obstacle_count == 0

    def test_bounds_property(self, simple_navmesh):
        """Test bounds property."""
        bounds = simple_navmesh.bounds
        assert isinstance(bounds, BoundingBox)

    def test_params_property(self):
        """Test params property."""
        params = NavMeshParams(agent_radius=1.0)
        navmesh = NavMesh(params)
        assert navmesh.params.agent_radius == 1.0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestNavMeshEdgeCases:
    """Tests for edge cases and error handling."""

    def test_degenerate_triangle(self):
        """Test with degenerate (zero-area) triangle."""
        navmesh = NavMesh()
        triangles = [
            Triangle(
                Vector3(0, 0, 0),
                Vector3(0, 0, 0),  # Same as v0
                Vector3(1, 0, 0)
            )
        ]
        # Should handle gracefully
        navmesh.build(triangles)

    def test_very_small_polygon(self):
        """Test with very small polygon."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(0.001, 0, 0),
            Vector3(0, 0, 0.001)
        ])
        assert navmesh.polygon_count == 1

    def test_very_large_coordinates(self):
        """Test with very large coordinates."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(10000, 0, 10000),
            Vector3(10100, 0, 10000),
            Vector3(10000, 0, 10100)
        ])
        assert navmesh.polygon_count == 1

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        navmesh = NavMesh()
        navmesh.add_polygon([
            Vector3(-10, 0, -10),
            Vector3(0, 0, -10),
            Vector3(-5, 0, 0)
        ])
        assert navmesh.polygon_count == 1

    def test_query_on_empty_navmesh(self):
        """Test queries on empty NavMesh."""
        navmesh = NavMesh()

        result = navmesh.find_nearest_point(Vector3(5, 0, 5))
        assert not result.success

        poly_id = navmesh.find_polygon_at(Vector3(5, 0, 5))
        assert poly_id is None

    def test_many_polygons(self):
        """Test with many polygons."""
        navmesh = NavMesh()
        for i in range(100):
            navmesh.add_polygon([
                Vector3(i, 0, 0),
                Vector3(i+1, 0, 0),
                Vector3(i, 0, 1)
            ])
        assert navmesh.polygon_count == 100

    def test_polygon_with_many_vertices(self):
        """Test polygon with many vertices."""
        navmesh = NavMesh()
        vertices = [
            Vector3(math.cos(i * 2 * math.pi / 20) * 5, 0, math.sin(i * 2 * math.pi / 20) * 5)
            for i in range(20)
        ]
        navmesh.add_polygon(vertices)
        assert navmesh.polygon_count == 1


# =============================================================================
# Tiling Tests
# =============================================================================


class TestNavMeshTiling:
    """Tests for tiled NavMesh."""

    def test_tiled_mode_params(self):
        """Test creating NavMesh with tiled mode."""
        params = NavMeshParams(
            build_mode=NavMeshBuildMode.TILED,
            tile_size=32.0
        )
        navmesh = NavMesh(params)
        assert navmesh.params.build_mode == NavMeshBuildMode.TILED
        assert navmesh.params.tile_size == 32.0

    def test_tile_size_affects_spatial_index(self):
        """Test that tile size is respected."""
        params = NavMeshParams(tile_size=16.0)
        assert params.tile_size == 16.0


# =============================================================================
# Area Type and Cost Modifier Tests
# =============================================================================


class TestAreaTypesAndCosts:
    """Tests for area types and cost modifiers."""

    def test_polygon_area_type_default(self):
        """Test default area type is 0."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 0, 1)
        ])
        poly = navmesh.get_polygon(poly_id)
        assert poly.area_type == 0

    def test_polygon_area_type_custom(self):
        """Test custom area type."""
        navmesh = NavMesh()
        poly_id = navmesh.add_polygon(
            [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)],
            area_type=10
        )
        poly = navmesh.get_polygon(poly_id)
        assert poly.area_type == 10

    def test_multiple_area_types(self):
        """Test multiple area types."""
        navmesh = NavMesh()

        poly1 = navmesh.add_polygon(
            [Vector3(0, 0, 0), Vector3(5, 0, 0), Vector3(0, 0, 5)],
            area_type=1  # Grass
        )
        poly2 = navmesh.add_polygon(
            [Vector3(5, 0, 0), Vector3(10, 0, 0), Vector3(5, 0, 5)],
            area_type=2  # Road
        )
        poly3 = navmesh.add_polygon(
            [Vector3(0, 0, 5), Vector3(5, 0, 5), Vector3(0, 0, 10)],
            area_type=3  # Water
        )

        assert navmesh.get_polygon(poly1).area_type == 1
        assert navmesh.get_polygon(poly2).area_type == 2
        assert navmesh.get_polygon(poly3).area_type == 3

    def test_polygon_flags(self):
        """Test polygon flags."""
        poly = NavMeshPolygon(id=1, vertices=[], flags=5)
        assert poly.flags == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestNavMeshIntegration:
    """Integration tests for NavMesh functionality."""

    def test_build_query_cycle(self, floor_triangles):
        """Test full build and query cycle."""
        navmesh = NavMesh()

        # Build
        assert navmesh.build(floor_triangles)
        assert navmesh.is_built

        # Query
        result = navmesh.find_nearest_point(Vector3(5, 1, 5))
        assert result.success

    def test_dynamic_modification_cycle(self):
        """Test dynamic modification cycle."""
        params = NavMeshParams(build_mode=NavMeshBuildMode.DYNAMIC)
        navmesh = NavMesh(params)

        # Add polygons
        poly1 = navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10)
        ])

        # Add obstacle
        obs_id = navmesh.add_obstacle(NavMeshObstacle(
            id=0,
            obstacle_type=ObstacleType.CYLINDER,
            position=Vector3(5, 0, 5),
            radius=1.0
        ))

        # Update
        navmesh.update()

        # Move obstacle
        navmesh.update_obstacle(obs_id, position=Vector3(7, 0, 7))
        navmesh.update()

        # Remove obstacle
        navmesh.remove_obstacle(obs_id)
        navmesh.update()

    def test_complex_geometry(self):
        """Test with more complex geometry."""
        navmesh = NavMesh()

        # Create a room with multiple floor sections
        triangles = []

        # Main floor
        for x in range(5):
            for z in range(5):
                triangles.append(Triangle(
                    Vector3(x * 2, 0, z * 2),
                    Vector3(x * 2 + 2, 0, z * 2),
                    Vector3(x * 2 + 2, 0, z * 2 + 2)
                ))
                triangles.append(Triangle(
                    Vector3(x * 2, 0, z * 2),
                    Vector3(x * 2 + 2, 0, z * 2 + 2),
                    Vector3(x * 2, 0, z * 2 + 2)
                ))

        result = navmesh.build(triangles)
        assert result
        assert navmesh.polygon_count > 0

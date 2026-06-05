"""Whitebox tests for closest point methods in SupportPolygon.

Tests internal implementation details of:
- closest_point_on_segment: Projects point onto line segment, clamped to bounds
- closest_point_on_boundary: Finds closest point on any polygon edge
- correction_vector: Returns vector to move outside point back to boundary

Task: T-FB-4.3 - Closest Point on Polygon
"""

from __future__ import annotations

import math
import pytest

from engine.animation.ik.fullbody import SupportPolygon
from engine.core.math.vec import Vec3


# =============================================================================
# Helper functions
# =============================================================================

def distance_xz(a: Vec3, b: Vec3) -> float:
    """Calculate distance in XZ plane, ignoring Y."""
    dx = a.x - b.x
    dz = a.z - b.z
    return math.sqrt(dx * dx + dz * dz)


def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-9) -> bool:
    """Check if two Vec3 are nearly equal."""
    return (
        abs(a.x - b.x) < eps and
        abs(a.y - b.y) < eps and
        abs(a.z - b.z) < eps
    )


# =============================================================================
# Tests for closest_point_on_segment (static method)
# =============================================================================

class TestClosestPointOnSegment:
    """Tests for the static closest_point_on_segment method.

    Implementation details:
    - Works in XZ plane only (Y is ignored)
    - Output always has Y=0 (ground plane)
    - Parameter t is clamped to [0, 1]
    - Degenerate segment (length < 1e-10) returns seg_start
    """

    def test_point_projects_to_middle(self) -> None:
        """Point perpendicular to segment middle projects to middle."""
        # Horizontal segment from (0,0,0) to (4,0,0)
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (2, 0, 2) - perpendicular to midpoint
        point = Vec3(2.0, 0.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 2.0) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 0.0) < 1e-9

    def test_point_projects_to_quarter(self) -> None:
        """Point projects to t=0.25 on segment."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (1, 0, 3) - perpendicular to t=0.25
        point = Vec3(1.0, 0.0, 3.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 1.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_point_clamps_to_start(self) -> None:
        """Point beyond start of segment clamps to seg_start."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (-2, 0, 1) - beyond the start
        point = Vec3(-2.0, 0.0, 1.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should clamp to t=0, returning seg_start coordinates
        assert abs(result.x - 0.0) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 0.0) < 1e-9

    def test_point_clamps_to_end(self) -> None:
        """Point beyond end of segment clamps to seg_end."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (6, 0, 1) - beyond the end
        point = Vec3(6.0, 0.0, 1.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should clamp to t=1, returning seg_end coordinates
        assert abs(result.x - 4.0) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 0.0) < 1e-9

    def test_degenerate_segment_returns_start(self) -> None:
        """Degenerate segment (zero length) returns seg_start."""
        seg_start = Vec3(2.0, 0.0, 3.0)
        seg_end = Vec3(2.0, 0.0, 3.0)  # Same as start
        point = Vec3(5.0, 0.0, 7.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 2.0) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 3.0) < 1e-9

    def test_near_degenerate_segment(self) -> None:
        """Segment with length < 1e-10 is treated as degenerate."""
        seg_start = Vec3(1.0, 0.0, 1.0)
        # Segment length = sqrt(1e-22) = 1e-11 < 1e-10 threshold
        seg_end = Vec3(1.0 + 1e-11, 0.0, 1.0)
        point = Vec3(5.0, 0.0, 5.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should return seg_start due to degenerate check
        assert abs(result.x - 1.0) < 1e-9
        assert abs(result.z - 1.0) < 1e-9

    def test_point_on_segment(self) -> None:
        """Point exactly on segment returns that point."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point exactly on segment at t=0.5
        point = Vec3(2.0, 0.0, 0.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_point_at_start(self) -> None:
        """Point exactly at segment start."""
        seg_start = Vec3(1.0, 0.0, 2.0)
        seg_end = Vec3(5.0, 0.0, 6.0)
        point = Vec3(1.0, 0.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 1.0) < 1e-9
        assert abs(result.z - 2.0) < 1e-9

    def test_point_at_end(self) -> None:
        """Point exactly at segment end."""
        seg_start = Vec3(1.0, 0.0, 2.0)
        seg_end = Vec3(5.0, 0.0, 6.0)
        point = Vec3(5.0, 0.0, 6.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 5.0) < 1e-9
        assert abs(result.z - 6.0) < 1e-9

    def test_diagonal_segment(self) -> None:
        """Test with diagonal segment (not axis-aligned)."""
        # Segment from (0,0,0) to (3,0,3) - 45 degree diagonal
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(3.0, 0.0, 3.0)
        # Point at (0, 0, 3) - perpendicular bisector intersection
        point = Vec3(0.0, 0.0, 3.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # t = ((0-0)*3 + (3-0)*3) / (9+9) = 9/18 = 0.5
        # Result should be at (1.5, 0, 1.5)
        assert abs(result.x - 1.5) < 1e-9
        assert abs(result.z - 1.5) < 1e-9

    def test_y_coordinate_ignored_in_point(self) -> None:
        """Y coordinate of query point is ignored."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (2, 100, 2) - Y is ignored
        point = Vec3(2.0, 100.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - 2.0) < 1e-9
        assert result.y == 0.0  # Output Y is always 0

    def test_y_coordinate_ignored_in_segment(self) -> None:
        """Y coordinates of segment are ignored."""
        seg_start = Vec3(0.0, 50.0, 0.0)  # Y=50
        seg_end = Vec3(4.0, -30.0, 0.0)   # Y=-30
        point = Vec3(2.0, 0.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should project to (2, 0, 0) - same as if Y were 0
        assert abs(result.x - 2.0) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 0.0) < 1e-9

    def test_output_always_on_ground_plane(self) -> None:
        """Output Y is always 0 (ground plane)."""
        seg_start = Vec3(0.0, 10.0, 0.0)
        seg_end = Vec3(4.0, 20.0, 4.0)
        point = Vec3(2.0, 100.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert result.y == 0.0

    def test_vertical_segment_in_z(self) -> None:
        """Segment vertical in Z direction (no X change)."""
        seg_start = Vec3(2.0, 0.0, 0.0)
        seg_end = Vec3(2.0, 0.0, 4.0)
        point = Vec3(5.0, 0.0, 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should project to (2, 0, 2)
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 2.0) < 1e-9

    def test_negative_coordinates(self) -> None:
        """Test with negative coordinates."""
        seg_start = Vec3(-4.0, 0.0, -2.0)
        seg_end = Vec3(0.0, 0.0, -2.0)
        point = Vec3(-2.0, 0.0, 0.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should project to (-2, 0, -2)
        assert abs(result.x - (-2.0)) < 1e-9
        assert abs(result.z - (-2.0)) < 1e-9

    def test_large_coordinates(self) -> None:
        """Test with large coordinate values."""
        large = 1e6
        seg_start = Vec3(large, 0.0, large)
        seg_end = Vec3(large + 4.0, 0.0, large)
        point = Vec3(large + 2.0, 0.0, large + 2.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert abs(result.x - (large + 2.0)) < 1e-6
        assert abs(result.z - large) < 1e-6

    def test_t_parameter_calculation(self) -> None:
        """Verify t parameter calculation for various positions."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(10.0, 0.0, 0.0)

        # Test t=0
        p0 = Vec3(0.0, 0.0, 5.0)
        r0 = SupportPolygon.closest_point_on_segment(p0, seg_start, seg_end)
        assert abs(r0.x - 0.0) < 1e-9

        # Test t=1
        p1 = Vec3(10.0, 0.0, 5.0)
        r1 = SupportPolygon.closest_point_on_segment(p1, seg_start, seg_end)
        assert abs(r1.x - 10.0) < 1e-9

        # Test t=0.3
        p03 = Vec3(3.0, 0.0, 5.0)
        r03 = SupportPolygon.closest_point_on_segment(p03, seg_start, seg_end)
        assert abs(r03.x - 3.0) < 1e-9

    def test_clamping_boundary_t_negative(self) -> None:
        """Verify clamping when t would be negative."""
        seg_start = Vec3(2.0, 0.0, 0.0)
        seg_end = Vec3(6.0, 0.0, 0.0)
        # Point at (0, 0, 1) - would give t = (0-2)*4 / 16 = -0.5
        point = Vec3(0.0, 0.0, 1.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Clamped to t=0, should return seg_start coords
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_clamping_boundary_t_greater_than_one(self) -> None:
        """Verify clamping when t would be > 1."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)
        # Point at (8, 0, 1) - would give t = 8*4 / 16 = 2.0
        point = Vec3(8.0, 0.0, 1.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Clamped to t=1, should return seg_end coords
        assert abs(result.x - 4.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9


# =============================================================================
# Tests for closest_point_on_boundary (instance method)
# =============================================================================

class TestClosestPointOnBoundary:
    """Tests for the closest_point_on_boundary method.

    Implementation details:
    - Iterates through all edges using modulo for wraparound
    - Returns Vec3.zero() for empty polygon
    - Returns single vertex if only one vertex exists
    - Finds minimum distance in XZ plane
    """

    def test_square_outside_point_to_right(self) -> None:
        """Point to the right of square finds closest on right edge."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        point = Vec3(4.0, 0.0, 1.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on right edge at (2, 0, 1)
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 1.0) < 1e-9

    def test_square_outside_point_above(self) -> None:
        """Point above square finds closest on top edge."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        point = Vec3(1.0, 0.0, 4.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on top edge at (1, 0, 2)
        assert abs(result.x - 1.0) < 1e-9
        assert abs(result.z - 2.0) < 1e-9

    def test_triangle_outside_point(self) -> None:
        """Point outside triangle."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 3.0),
        ])
        # Point to the right of triangle
        point = Vec3(5.0, 0.0, 1.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on edge from (4,0,0) to (2,0,3)
        # Verify it's on the boundary by checking it's closer than vertices
        dist_to_result = distance_xz(point, result)
        dist_to_v0 = distance_xz(point, Vec3(4.0, 0.0, 0.0))
        assert dist_to_result <= dist_to_v0 + 1e-9

    def test_closest_to_vertex_corner(self) -> None:
        """Point closest to a vertex (corner) of polygon."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Point diagonal from corner
        point = Vec3(-1.0, 0.0, -1.0)

        result = polygon.closest_point_on_boundary(point)

        # Closest point should be at corner (0, 0, 0)
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_closest_to_edge_middle(self) -> None:
        """Point perpendicular to edge middle."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point perpendicular to bottom edge at midpoint
        point = Vec3(2.0, 0.0, -2.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be at (2, 0, 0) - middle of bottom edge
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_point_inside_polygon(self) -> None:
        """Point inside polygon still finds closest boundary point."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point inside, closer to bottom edge
        point = Vec3(2.0, 0.0, 0.5)

        result = polygon.closest_point_on_boundary(point)

        # Should be on bottom edge at (2, 0, 0)
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_empty_polygon_returns_zero(self) -> None:
        """Empty polygon returns Vec3.zero()."""
        polygon = SupportPolygon(vertices=[])
        point = Vec3(1.0, 0.0, 1.0)

        result = polygon.closest_point_on_boundary(point)

        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_single_vertex_returns_vertex(self) -> None:
        """Single vertex polygon returns that vertex."""
        polygon = SupportPolygon(vertices=[Vec3(3.0, 0.0, 4.0)])
        point = Vec3(10.0, 0.0, 10.0)

        result = polygon.closest_point_on_boundary(point)

        assert abs(result.x - 3.0) < 1e-9
        assert abs(result.z - 4.0) < 1e-9

    def test_two_vertices_line_segment(self) -> None:
        """Two vertex polygon is treated as line segment."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
        ])
        # Point perpendicular to line
        point = Vec3(2.0, 0.0, 3.0)

        result = polygon.closest_point_on_boundary(point)

        # Should project to (2, 0, 0) on the "edge" from v0 to v1
        # And the wraparound edge from v1 to v0
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_wraparound_edge(self) -> None:
        """Test edge from last vertex to first vertex."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Point closer to left edge (wraparound edge from v3 to v0)
        point = Vec3(-1.0, 0.0, 1.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on left edge at (0, 0, 1)
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - 1.0) < 1e-9

    def test_equidistant_from_two_edges(self) -> None:
        """Point equidistant from two edges returns closest boundary point."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Point at corner extension (equidistant from two edges)
        point = Vec3(-1.0, 0.0, -1.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be at corner (0, 0, 0) - where two edges meet
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9
        # Distance from query point to corner is sqrt(2)
        dist = distance_xz(point, result)
        assert abs(dist - math.sqrt(2.0)) < 1e-9

    def test_concave_polygon(self) -> None:
        """Test with concave (non-convex) polygon."""
        # L-shaped polygon
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Point in the "notch" area
        point = Vec3(1.5, 0.0, 1.5)

        result = polygon.closest_point_on_boundary(point)

        # Should be closest to edge from (1,0,1) to (1,0,2) at (1, 0, 1.5)
        # or edge from (2,0,1) to (1,0,1)
        dist = distance_xz(point, result)
        assert dist < 1.0  # Should be reasonably close

    def test_y_coordinate_ignored(self) -> None:
        """Y coordinate of point is ignored in XZ plane calculation."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        point = Vec3(3.0, 100.0, 1.0)  # Y=100

        result = polygon.closest_point_on_boundary(point)

        # Y should be 0 in result
        assert result.y == 0.0

    def test_output_on_ground_plane(self) -> None:
        """Output always has Y=0."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 5.0, 0.0),   # Y values in vertices (shouldn't matter)
            Vec3(2.0, 10.0, 0.0),
            Vec3(1.0, -3.0, 2.0),
        ])
        point = Vec3(3.0, 0.0, 1.0)

        result = polygon.closest_point_on_boundary(point)

        assert result.y == 0.0

    def test_many_edges_finds_closest(self) -> None:
        """Polygon with many edges finds the truly closest one."""
        # Octagon
        vertices = []
        for i in range(8):
            angle = i * math.pi / 4
            vertices.append(Vec3(2.0 * math.cos(angle), 0.0, 2.0 * math.sin(angle)))
        polygon = SupportPolygon(vertices=vertices)

        # Point clearly closest to one edge
        point = Vec3(5.0, 0.0, 0.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on right edge, approximately at (2, 0, 0)
        # The octagon vertex at angle=0 is at (2, 0, 0)
        assert result.x > 1.5  # Should be on right side


# =============================================================================
# Tests for correction_vector (instance method)
# =============================================================================

class TestCorrectionVector:
    """Tests for the correction_vector method.

    Implementation details:
    - Returns Vec3.zero() if point is inside polygon
    - Returns vector from point to closest boundary point if outside
    - Uses contains_point for inside check
    - Uses closest_point_on_boundary for finding target
    - Y component of result is always 0
    """

    def test_inside_returns_zero(self) -> None:
        """Point inside polygon returns zero vector."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point clearly inside
        point = Vec3(2.0, 0.0, 2.0)

        result = polygon.correction_vector(point)

        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_inside_near_edge_returns_zero(self) -> None:
        """Point inside but close to edge still returns zero."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point very close to edge but inside
        point = Vec3(2.0, 0.0, 0.001)

        result = polygon.correction_vector(point)

        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_outside_returns_correction(self) -> None:
        """Point outside polygon returns vector toward boundary."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point outside to the right
        point = Vec3(6.0, 0.0, 2.0)

        result = polygon.correction_vector(point)

        # Should point toward (4, 0, 2) which is on boundary
        # Correction = closest - point = (4-6, 0, 2-2) = (-2, 0, 0)
        assert abs(result.x - (-2.0)) < 1e-9
        assert result.y == 0.0
        assert abs(result.z - 0.0) < 1e-9

    def test_correction_magnitude(self) -> None:
        """Correction vector magnitude equals distance to boundary."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point 3 units outside to the right
        point = Vec3(7.0, 0.0, 2.0)

        result = polygon.correction_vector(point)

        # Distance to boundary at (4, 0, 2) should be 3
        magnitude = math.sqrt(result.x * result.x + result.z * result.z)
        assert abs(magnitude - 3.0) < 1e-9

    def test_correction_direction_toward_boundary(self) -> None:
        """Correction vector points toward nearest boundary."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point below polygon
        point = Vec3(2.0, 0.0, -2.0)

        result = polygon.correction_vector(point)

        # Should point toward (2, 0, 0) - positive Z direction
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - 2.0) < 1e-9

    def test_correction_plus_point_equals_boundary(self) -> None:
        """Point + correction_vector = closest boundary point."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        point = Vec3(6.0, 0.0, 2.0)

        correction = polygon.correction_vector(point)
        closest = polygon.closest_point_on_boundary(point)

        # point + correction should equal closest
        corrected = Vec3(point.x + correction.x, point.y + correction.y, point.z + correction.z)
        assert abs(corrected.x - closest.x) < 1e-9
        assert abs(corrected.z - closest.z) < 1e-9

    def test_corner_case_outside(self) -> None:
        """Point outside near corner."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        # Point diagonally outside corner
        point = Vec3(-1.0, 0.0, -1.0)

        result = polygon.correction_vector(point)

        # Should point toward corner (0, 0, 0)
        # Correction = (0-(-1), 0, 0-(-1)) = (1, 0, 1)
        assert abs(result.x - 1.0) < 1e-9
        assert abs(result.z - 1.0) < 1e-9

    def test_y_coordinate_zero_in_result(self) -> None:
        """Correction vector Y component is always 0."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 4.0),
        ])
        point = Vec3(5.0, 50.0, 2.0)  # Y=50

        result = polygon.correction_vector(point)

        assert result.y == 0.0

    def test_empty_polygon(self) -> None:
        """Empty polygon - contains_point returns False, closest returns zero."""
        polygon = SupportPolygon(vertices=[])
        point = Vec3(1.0, 0.0, 1.0)

        result = polygon.correction_vector(point)

        # closest_point_on_boundary returns zero for empty
        # correction = zero - point = (-1, 0, -1)
        assert abs(result.x - (-1.0)) < 1e-9
        assert abs(result.z - (-1.0)) < 1e-9

    def test_single_vertex_polygon(self) -> None:
        """Single vertex polygon - point is never 'inside'."""
        polygon = SupportPolygon(vertices=[Vec3(2.0, 0.0, 3.0)])
        point = Vec3(5.0, 0.0, 7.0)

        result = polygon.correction_vector(point)

        # contains_point returns False for < 3 vertices
        # So we get correction toward the single vertex
        assert abs(result.x - (2.0 - 5.0)) < 1e-9
        assert abs(result.z - (3.0 - 7.0)) < 1e-9

    def test_two_vertex_polygon(self) -> None:
        """Two vertex polygon - degenerate line."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
        ])
        point = Vec3(2.0, 0.0, 2.0)

        result = polygon.correction_vector(point)

        # contains_point returns False, so we get correction
        # closest point is (2, 0, 0), correction = (0, 0, -2)
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - (-2.0)) < 1e-9

    def test_triangle_outside(self) -> None:
        """Point outside triangle."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 3.0),
        ])
        # Point clearly outside
        point = Vec3(2.0, 0.0, -2.0)

        result = polygon.correction_vector(point)

        # Closest is on bottom edge at (2, 0, 0)
        # Correction = (2-2, 0, 0-(-2)) = (0, 0, 2)
        assert abs(result.x - 0.0) < 1e-9
        assert abs(result.z - 2.0) < 1e-9

    def test_triangle_inside(self) -> None:
        """Point inside triangle returns zero."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 3.0),
        ])
        # Point at centroid
        point = Vec3(2.0, 0.0, 1.0)

        result = polygon.correction_vector(point)

        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0


# =============================================================================
# Integration tests - method interactions
# =============================================================================

class TestMethodInteractions:
    """Tests verifying correct interaction between the methods."""

    def test_segment_used_by_boundary(self) -> None:
        """closest_point_on_boundary uses closest_point_on_segment internally."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
        ])
        point = Vec3(2.0, 0.0, 3.0)

        # Get result from boundary method
        boundary_result = polygon.closest_point_on_boundary(point)

        # Manually call segment method on the single edge
        segment_result = SupportPolygon.closest_point_on_segment(
            point,
            polygon.vertices[0],
            polygon.vertices[1]
        )

        # They should match (considering wraparound edge gives same result)
        assert abs(boundary_result.x - segment_result.x) < 1e-9
        assert abs(boundary_result.z - segment_result.z) < 1e-9

    def test_correction_uses_contains_and_boundary(self) -> None:
        """correction_vector uses contains_point and closest_point_on_boundary."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 4.0),
            Vec3(0.0, 0.0, 4.0),
        ])
        point = Vec3(5.0, 0.0, 2.0)

        # Check that contains_point says outside
        assert polygon.contains_point(point) is False

        # Get closest boundary point
        closest = polygon.closest_point_on_boundary(point)

        # Get correction vector
        correction = polygon.correction_vector(point)

        # Verify correction = closest - point
        assert abs(correction.x - (closest.x - point.x)) < 1e-9
        assert abs(correction.z - (closest.z - point.z)) < 1e-9

    def test_all_methods_work_with_same_polygon(self) -> None:
        """All three methods work correctly on the same polygon instance."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
            Vec3(3.0, 0.0, 3.0),
            Vec3(0.0, 0.0, 3.0),
        ])

        # Point outside
        outside = Vec3(5.0, 0.0, 1.5)

        # Test segment method (static, but can use with polygon edges)
        seg_closest = SupportPolygon.closest_point_on_segment(
            outside,
            polygon.vertices[1],  # (3,0,0)
            polygon.vertices[2],  # (3,0,3)
        )
        assert abs(seg_closest.x - 3.0) < 1e-9
        assert abs(seg_closest.z - 1.5) < 1e-9

        # Test boundary method
        boundary_closest = polygon.closest_point_on_boundary(outside)
        assert abs(boundary_closest.x - 3.0) < 1e-9
        assert abs(boundary_closest.z - 1.5) < 1e-9

        # Test correction method
        correction = polygon.correction_vector(outside)
        assert abs(correction.x - (-2.0)) < 1e-9
        assert abs(correction.z - 0.0) < 1e-9


# =============================================================================
# Edge cases and numerical stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical edge cases and stability."""

    def test_very_small_polygon(self) -> None:
        """Polygon with very small edge lengths."""
        epsilon = 1e-6
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(epsilon, 0.0, 0.0),
            Vec3(epsilon / 2, 0.0, epsilon),
        ])
        point = Vec3(epsilon * 2, 0.0, epsilon / 2)

        result = polygon.closest_point_on_boundary(point)

        # Should not crash and return reasonable result
        assert isinstance(result, Vec3)
        assert not math.isnan(result.x)
        assert not math.isnan(result.z)

    def test_very_large_polygon(self) -> None:
        """Polygon with very large coordinate values."""
        large = 1e9
        polygon = SupportPolygon(vertices=[
            Vec3(large, 0.0, large),
            Vec3(large + 4.0, 0.0, large),
            Vec3(large + 2.0, 0.0, large + 3.0),
        ])
        point = Vec3(large + 2.0, 0.0, large - 1.0)

        result = polygon.closest_point_on_boundary(point)

        # Should be on bottom edge
        assert abs(result.x - (large + 2.0)) < 1e-3
        assert abs(result.z - large) < 1e-3

    def test_point_very_close_to_edge(self) -> None:
        """Point extremely close to edge."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 4.0),
        ])
        # Point very slightly outside bottom edge
        point = Vec3(2.0, 0.0, -1e-12)

        result = polygon.closest_point_on_boundary(point)

        # Should be at (2, 0, 0)
        assert abs(result.x - 2.0) < 1e-9
        assert abs(result.z - 0.0) < 1e-9

    def test_segment_with_near_zero_length(self) -> None:
        """Segment with length very close to degenerate threshold."""
        # Length = sqrt(1e-20) = 1e-10, exactly at threshold
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(1e-10, 0.0, 0.0)
        point = Vec3(1.0, 0.0, 1.0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        # Should not crash
        assert isinstance(result, Vec3)
        assert not math.isnan(result.x)

    def test_collinear_polygon_vertices(self) -> None:
        """Polygon where all vertices are collinear (degenerate)."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
        ])
        point = Vec3(2.0, 0.0, 2.0)

        result = polygon.closest_point_on_boundary(point)

        # Should find closest point on the line
        assert abs(result.z - 0.0) < 1e-9
        assert 0.0 <= result.x <= 4.0


# =============================================================================
# Performance-oriented tests (structure, not timing)
# =============================================================================

class TestAlgorithmStructure:
    """Tests verifying algorithm structure without timing."""

    def test_boundary_iterates_all_edges(self) -> None:
        """Verify boundary method checks all edges by testing specific points."""
        # Pentagon - each edge is closest for points in different directions
        import math
        vertices = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 2  # Start from top
            vertices.append(Vec3(math.cos(angle), 0.0, math.sin(angle)))
        polygon = SupportPolygon(vertices=vertices)

        # Test point closest to each edge
        for i in range(5):
            # Point in direction of edge midpoint
            mid_angle = (i + 0.5) * 2 * math.pi / 5 + math.pi / 2
            point = Vec3(3.0 * math.cos(mid_angle), 0.0, 3.0 * math.sin(mid_angle))

            result = polygon.closest_point_on_boundary(point)

            # Result should be closer to point than any vertex
            dist_to_result = distance_xz(point, result)
            for v in vertices:
                dist_to_vertex = distance_xz(point, v)
                assert dist_to_result <= dist_to_vertex + 1e-9

    def test_segment_uses_dot_product(self) -> None:
        """Verify segment method uses dot product projection correctly."""
        seg_start = Vec3(0.0, 0.0, 0.0)
        seg_end = Vec3(4.0, 0.0, 0.0)

        # For point perpendicular to segment, result should be projection
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            proj_x = t * 4.0
            point = Vec3(proj_x, 0.0, 3.0)  # Perpendicular distance doesn't matter

            result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

            assert abs(result.x - proj_x) < 1e-9
            assert abs(result.z - 0.0) < 1e-9

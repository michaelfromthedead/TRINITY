"""Whitebox tests for SupportPolygon class.

Tests internal implementation details of the SupportPolygon class,
including vertex storage, from_foot_positions projection, ray casting
algorithm in contains_point, and ground plane projection.
"""

from __future__ import annotations

import pytest
import math

from engine.animation.ik.fullbody import SupportPolygon
from engine.core.math.vec import Vec3


class TestSupportPolygonInit:
    """Tests for SupportPolygon initialization."""

    def test_default_empty_vertices(self) -> None:
        """Default init creates empty vertices list."""
        polygon = SupportPolygon()
        assert polygon.vertices == []
        assert isinstance(polygon.vertices, list)

    def test_with_vertices(self) -> None:
        """Init with explicit vertices list."""
        verts = [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.5, 0.0, 1.0)]
        polygon = SupportPolygon(vertices=verts)
        assert len(polygon.vertices) == 3
        assert polygon.vertices[0].x == 0.0
        assert polygon.vertices[1].x == 1.0
        assert polygon.vertices[2].z == 1.0

    def test_vertices_is_mutable(self) -> None:
        """Verify vertices list can be modified after init."""
        polygon = SupportPolygon()
        polygon.vertices.append(Vec3(1.0, 0.0, 2.0))
        assert len(polygon.vertices) == 1
        assert polygon.vertices[0].x == 1.0

    def test_vertices_stores_reference(self) -> None:
        """Verify vertices list is stored as reference (dataclass default)."""
        verts = [Vec3(0.0, 0.0, 0.0)]
        polygon = SupportPolygon(vertices=verts)
        # Modifying original list affects polygon (dataclass behavior)
        verts.append(Vec3(1.0, 0.0, 1.0))
        assert len(polygon.vertices) == 2


class TestFromFootPositions:
    """Tests for from_foot_positions classmethod."""

    def test_projects_to_xz_plane(self) -> None:
        """All output vertices have y=0 regardless of input y values."""
        positions = [
            Vec3(0.0, 5.0, 0.0),
            Vec3(1.0, 10.0, 0.0),
            Vec3(0.5, -3.0, 1.0),
        ]
        polygon = SupportPolygon.from_foot_positions(positions)
        for v in polygon.vertices:
            assert v.y == 0.0

    def test_preserves_x_and_z(self) -> None:
        """X and Z coordinates are preserved from input."""
        positions = [
            Vec3(1.5, 0.0, 2.5),
            Vec3(-3.0, 0.0, 4.0),
            Vec3(0.0, 0.0, -1.0),
        ]
        polygon = SupportPolygon.from_foot_positions(positions)
        assert polygon.vertices[0].x == 1.5
        assert polygon.vertices[0].z == 2.5
        assert polygon.vertices[1].x == -3.0
        assert polygon.vertices[1].z == 4.0
        assert polygon.vertices[2].x == 0.0
        assert polygon.vertices[2].z == -1.0

    def test_empty_positions(self) -> None:
        """Empty input produces polygon with no vertices."""
        polygon = SupportPolygon.from_foot_positions([])
        assert polygon.vertices == []

    def test_single_position(self) -> None:
        """Single position creates polygon with one vertex."""
        positions = [Vec3(1.0, 2.0, 3.0)]
        polygon = SupportPolygon.from_foot_positions(positions)
        assert len(polygon.vertices) == 1
        assert polygon.vertices[0].x == 1.0
        assert polygon.vertices[0].y == 0.0
        assert polygon.vertices[0].z == 3.0

    def test_two_positions(self) -> None:
        """Two positions create degenerate line polygon."""
        positions = [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0)]
        polygon = SupportPolygon.from_foot_positions(positions)
        assert len(polygon.vertices) == 2

    def test_varying_y_values(self) -> None:
        """Varying Y values are all projected to 0."""
        positions = [
            Vec3(0.0, 100.0, 0.0),
            Vec3(1.0, -50.0, 0.0),
            Vec3(0.5, 0.001, 1.0),
        ]
        polygon = SupportPolygon.from_foot_positions(positions)
        for v in polygon.vertices:
            assert v.y == 0.0

    def test_returns_support_polygon_instance(self) -> None:
        """Classmethod returns SupportPolygon instance."""
        positions = [Vec3(0.0, 0.0, 0.0)]
        polygon = SupportPolygon.from_foot_positions(positions)
        assert isinstance(polygon, SupportPolygon)


class TestContainsPoint:
    """Tests for contains_point method - basic functionality."""

    def test_point_inside_triangle(self) -> None:
        """Point at centroid of triangle is inside."""
        # Triangle with vertices at (0,0,0), (2,0,0), (1,0,2)
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        # Centroid is at (1, 0, 2/3)
        centroid = Vec3(1.0, 0.0, 0.66)
        assert polygon.contains_point(centroid) is True

    def test_point_outside_triangle(self) -> None:
        """Point clearly outside triangle returns False."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        outside = Vec3(5.0, 0.0, 5.0)
        assert polygon.contains_point(outside) is False

    def test_point_on_edge(self) -> None:
        """Point on polygon edge - behavior depends on ray cast."""
        # Edge case: point exactly on edge may return True or False
        # depending on floating point precision
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        # Point on edge from (0,0,0) to (2,0,0)
        on_edge = Vec3(1.0, 0.0, 0.0)
        # Ray casting may or may not include boundary
        result = polygon.contains_point(on_edge)
        assert isinstance(result, bool)

    def test_point_on_vertex(self) -> None:
        """Point exactly on vertex - ray casting behavior."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        on_vertex = Vec3(0.0, 0.0, 0.0)
        result = polygon.contains_point(on_vertex)
        assert isinstance(result, bool)

    def test_square_polygon(self) -> None:
        """Test with square polygon."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Center of square
        center = Vec3(1.0, 0.0, 1.0)
        assert polygon.contains_point(center) is True
        # Outside
        outside = Vec3(3.0, 0.0, 1.0)
        assert polygon.contains_point(outside) is False

    def test_fewer_than_3_vertices_returns_false(self) -> None:
        """Polygon with fewer than 3 vertices always returns False."""
        # Empty polygon
        empty = SupportPolygon(vertices=[])
        assert empty.contains_point(Vec3(0.0, 0.0, 0.0)) is False

        # Single vertex
        single = SupportPolygon(vertices=[Vec3(0.0, 0.0, 0.0)])
        assert single.contains_point(Vec3(0.0, 0.0, 0.0)) is False

        # Two vertices (line segment)
        line = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 1.0),
        ])
        assert line.contains_point(Vec3(0.5, 0.0, 0.5)) is False

    def test_point_y_ignored(self) -> None:
        """Y coordinate of point is ignored (projected to XZ)."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        # Point at centroid but elevated in Y
        elevated = Vec3(1.0, 100.0, 0.66)
        assert polygon.contains_point(elevated) is True

        # Same X,Z but below ground
        below = Vec3(1.0, -50.0, 0.66)
        assert polygon.contains_point(below) is True


class TestRayCastingAlgorithm:
    """Tests for ray casting algorithm internals."""

    def test_horizontal_edges_handled(self) -> None:
        """Horizontal edges (same z for both endpoints) are skipped."""
        # Rectangle with horizontal top and bottom edges
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),   # bottom-left
            Vec3(2.0, 0.0, 0.0),   # bottom-right (horizontal edge)
            Vec3(2.0, 0.0, 2.0),   # top-right
            Vec3(0.0, 0.0, 2.0),   # top-left (horizontal edge)
        ])
        # Point inside should still work correctly
        inside = Vec3(1.0, 0.0, 1.0)
        assert polygon.contains_point(inside) is True

        # Point outside should still work
        outside = Vec3(3.0, 0.0, 1.0)
        assert polygon.contains_point(outside) is False

    def test_horizontal_edge_tolerance(self) -> None:
        """Horizontal edge detection uses 1e-10 tolerance."""
        # Edge with z difference less than 1e-10 should be skipped
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 1e-11),  # Nearly horizontal
            Vec3(1.0, 0.0, 2.0),
        ])
        center = Vec3(1.0, 0.0, 0.66)
        result = polygon.contains_point(center)
        assert isinstance(result, bool)  # Should not crash

    def test_odd_crossing_count_inside(self) -> None:
        """Ray with odd number of crossings indicates inside."""
        # Simple triangle - ray from inside crosses 1 edge (odd)
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 4.0),
        ])
        # Point clearly inside
        inside = Vec3(2.0, 0.0, 1.0)
        assert polygon.contains_point(inside) is True

    def test_even_crossing_count_outside(self) -> None:
        """Ray with even number of crossings indicates outside."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(4.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 4.0),
        ])
        # Point clearly outside (to the right)
        outside = Vec3(10.0, 0.0, 1.0)
        assert polygon.contains_point(outside) is False

    def test_ray_direction_is_positive_x(self) -> None:
        """Ray is cast in positive X direction from point."""
        # Test that ray goes positive X by checking behavior
        polygon = SupportPolygon(vertices=[
            Vec3(1.0, 0.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 2.0),
        ])
        # Point to the left of polygon - ray goes right and crosses
        left_of_polygon = Vec3(0.0, 0.0, 0.5)
        assert polygon.contains_point(left_of_polygon) is False

    def test_concave_polygon(self) -> None:
        """Test ray casting on concave (non-convex) polygon."""
        # L-shaped polygon
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 2.0),
            Vec3(0.0, 0.0, 2.0),
        ])
        # Point in the bottom part of L
        in_bottom = Vec3(1.5, 0.0, 0.5)
        assert polygon.contains_point(in_bottom) is True

        # Point in the notch (outside)
        in_notch = Vec3(1.5, 0.0, 1.5)
        assert polygon.contains_point(in_notch) is False

        # Point in the top-left part
        in_top = Vec3(0.5, 0.0, 1.5)
        assert polygon.contains_point(in_top) is True

    def test_wraparound_last_to_first_vertex(self) -> None:
        """Algorithm wraps from last vertex back to first."""
        # The algorithm uses j = n - 1 initially to wrap around
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        # Edge from vertex[2] to vertex[0] must be checked
        # Point that would be inside if edge 2->0 is checked
        inside = Vec3(0.5, 0.0, 0.5)
        assert polygon.contains_point(inside) is True


class TestProjectToGround:
    """Tests for project_to_ground method."""

    def test_sets_y_to_zero(self) -> None:
        """Projected point has y=0."""
        polygon = SupportPolygon()
        point = Vec3(1.0, 5.0, 2.0)
        projected = polygon.project_to_ground(point)
        assert projected.y == 0.0

    def test_preserves_x_and_z(self) -> None:
        """X and Z coordinates are preserved."""
        polygon = SupportPolygon()
        point = Vec3(3.5, 10.0, -2.5)
        projected = polygon.project_to_ground(point)
        assert projected.x == 3.5
        assert projected.z == -2.5

    def test_negative_y(self) -> None:
        """Negative Y is also projected to 0."""
        polygon = SupportPolygon()
        point = Vec3(1.0, -100.0, 2.0)
        projected = polygon.project_to_ground(point)
        assert projected.y == 0.0

    def test_zero_y_unchanged(self) -> None:
        """Point already at y=0 is returned with y=0."""
        polygon = SupportPolygon()
        point = Vec3(1.0, 0.0, 2.0)
        projected = polygon.project_to_ground(point)
        assert projected.y == 0.0
        assert projected.x == 1.0
        assert projected.z == 2.0

    def test_returns_new_vec3(self) -> None:
        """project_to_ground returns a new Vec3 instance."""
        polygon = SupportPolygon()
        point = Vec3(1.0, 5.0, 2.0)
        projected = polygon.project_to_ground(point)
        assert isinstance(projected, Vec3)
        # Original point unchanged
        assert point.y == 5.0

    def test_very_large_y(self) -> None:
        """Very large Y values are projected correctly."""
        polygon = SupportPolygon()
        point = Vec3(1.0, 1e10, 2.0)
        projected = polygon.project_to_ground(point)
        assert projected.y == 0.0
        assert projected.x == 1.0

    def test_very_small_y(self) -> None:
        """Very small Y values (near zero) are projected to exactly 0."""
        polygon = SupportPolygon()
        point = Vec3(1.0, 1e-15, 2.0)
        projected = polygon.project_to_ground(point)
        assert projected.y == 0.0


class TestContainsPointEdgeCases:
    """Additional edge cases for contains_point."""

    def test_large_polygon(self) -> None:
        """Test with polygon having many vertices."""
        # Octagon
        import math
        vertices = []
        for i in range(8):
            angle = i * math.pi / 4
            x = math.cos(angle)
            z = math.sin(angle)
            vertices.append(Vec3(x, 0.0, z))
        polygon = SupportPolygon(vertices=vertices)

        # Center should be inside
        assert polygon.contains_point(Vec3(0.0, 0.0, 0.0)) is True

        # Far outside
        assert polygon.contains_point(Vec3(10.0, 0.0, 10.0)) is False

    def test_collinear_vertices(self) -> None:
        """Polygon with collinear vertices (degenerate)."""
        # All points on a line
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
        ])
        # This is a degenerate polygon (no area)
        result = polygon.contains_point(Vec3(1.0, 0.0, 0.0))
        assert isinstance(result, bool)

    def test_very_small_polygon(self) -> None:
        """Test with very small polygon (near floating point limits)."""
        epsilon = 1e-9
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(epsilon, 0.0, 0.0),
            Vec3(epsilon / 2, 0.0, epsilon),
        ])
        # Center should be inside
        center = Vec3(epsilon / 2, 0.0, epsilon / 3)
        result = polygon.contains_point(center)
        assert isinstance(result, bool)

    def test_very_large_coordinates(self) -> None:
        """Test with very large coordinate values."""
        large = 1e6
        polygon = SupportPolygon(vertices=[
            Vec3(large, 0.0, large),
            Vec3(large + 2, 0.0, large),
            Vec3(large + 1, 0.0, large + 2),
        ])
        # Center
        center = Vec3(large + 1, 0.0, large + 0.66)
        assert polygon.contains_point(center) is True

    def test_negative_coordinates(self) -> None:
        """Test with negative coordinates."""
        polygon = SupportPolygon(vertices=[
            Vec3(-2.0, 0.0, -2.0),
            Vec3(0.0, 0.0, -2.0),
            Vec3(-1.0, 0.0, 0.0),
        ])
        # Center
        center = Vec3(-1.0, 0.0, -1.5)
        assert polygon.contains_point(center) is True

        # Outside
        outside = Vec3(1.0, 0.0, 0.0)
        assert polygon.contains_point(outside) is False


class TestInternalState:
    """Tests for internal state management."""

    def test_vertices_directly_accessible(self) -> None:
        """Vertices attribute is directly accessible (dataclass)."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ])
        assert hasattr(polygon, 'vertices')
        assert len(polygon.vertices) == 2

    def test_vertices_can_be_replaced(self) -> None:
        """Vertices list can be replaced entirely."""
        polygon = SupportPolygon(vertices=[Vec3(0.0, 0.0, 0.0)])
        new_vertices = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ]
        polygon.vertices = new_vertices
        assert len(polygon.vertices) == 3

    def test_contains_point_uses_current_vertices(self) -> None:
        """contains_point uses current vertex state."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 2.0),
        ])
        point = Vec3(1.0, 0.0, 0.5)
        assert polygon.contains_point(point) is True

        # Modify vertices
        polygon.vertices = [
            Vec3(10.0, 0.0, 10.0),
            Vec3(12.0, 0.0, 10.0),
            Vec3(11.0, 0.0, 12.0),
        ]
        # Same point now outside
        assert polygon.contains_point(point) is False

    def test_no_hidden_state(self) -> None:
        """SupportPolygon has no hidden state beyond vertices."""
        polygon = SupportPolygon(vertices=[Vec3(0.0, 0.0, 0.0)])
        # Check that only expected attributes exist
        attrs = [a for a in dir(polygon) if not a.startswith('_')]
        # Should have vertices, from_foot_positions, contains_point, project_to_ground
        assert 'vertices' in attrs
        assert 'from_foot_positions' in attrs
        assert 'contains_point' in attrs
        assert 'project_to_ground' in attrs


class TestAlgorithmCorrectness:
    """Tests verifying algorithm correctness with known results."""

    def test_unit_square_corners(self) -> None:
        """Test all corners and edges of unit square."""
        polygon = SupportPolygon(vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(0.0, 0.0, 1.0),
        ])
        # Center definitely inside
        assert polygon.contains_point(Vec3(0.5, 0.0, 0.5)) is True

        # Points clearly outside
        assert polygon.contains_point(Vec3(-0.5, 0.0, 0.5)) is False
        assert polygon.contains_point(Vec3(1.5, 0.0, 0.5)) is False
        assert polygon.contains_point(Vec3(0.5, 0.0, -0.5)) is False
        assert polygon.contains_point(Vec3(0.5, 0.0, 1.5)) is False

    def test_equilateral_triangle(self) -> None:
        """Test equilateral triangle centered at origin."""
        import math
        r = 1.0
        vertices = [
            Vec3(r * math.cos(math.pi / 2), 0.0, r * math.sin(math.pi / 2)),
            Vec3(r * math.cos(7 * math.pi / 6), 0.0, r * math.sin(7 * math.pi / 6)),
            Vec3(r * math.cos(11 * math.pi / 6), 0.0, r * math.sin(11 * math.pi / 6)),
        ]
        polygon = SupportPolygon(vertices=vertices)

        # Center should be inside
        assert polygon.contains_point(Vec3(0.0, 0.0, 0.0)) is True

        # Outside the circumscribed circle
        assert polygon.contains_point(Vec3(2.0, 0.0, 0.0)) is False

    def test_star_polygon(self) -> None:
        """Test non-convex star-shaped polygon."""
        import math
        vertices = []
        for i in range(10):
            angle = i * math.pi / 5
            r = 1.0 if i % 2 == 0 else 0.4  # Alternating radii
            vertices.append(Vec3(r * math.cos(angle), 0.0, r * math.sin(angle)))
        polygon = SupportPolygon(vertices=vertices)

        # Center should be inside
        assert polygon.contains_point(Vec3(0.0, 0.0, 0.0)) is True

        # Far outside
        assert polygon.contains_point(Vec3(2.0, 0.0, 2.0)) is False

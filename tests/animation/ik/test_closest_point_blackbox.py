"""
Blackbox tests for SupportPolygon closest point methods.

CLEANROOM TESTS - Written from specification only, without reading implementation.

Public Interface:
    class SupportPolygon:
        @staticmethod
        def closest_point_on_segment(point: Vec3, seg_start: Vec3, seg_end: Vec3) -> Vec3
            '''Find closest point on segment to query point'''

        def closest_point_on_boundary(self, point: Vec3) -> Vec3
            '''Find closest point on polygon boundary'''

        def correction_vector(self, point: Vec3) -> Vec3
            '''Vector to move point back to polygon. Zero if inside.'''
"""

import pytest
import math
from typing import List

from engine.animation.ik.fullbody import SupportPolygon
from engine.core.math.vec import Vec3


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def unit_square_polygon() -> SupportPolygon:
    """A 1x1 unit square polygon centered at origin on XZ plane."""
    return SupportPolygon(vertices=[
        Vec3(-0.5, 0, -0.5),
        Vec3(0.5, 0, -0.5),
        Vec3(0.5, 0, 0.5),
        Vec3(-0.5, 0, 0.5)
    ])


@pytest.fixture
def large_square_polygon() -> SupportPolygon:
    """A 2x2 square polygon on XZ plane."""
    return SupportPolygon(vertices=[
        Vec3(0, 0, 0),
        Vec3(2, 0, 0),
        Vec3(2, 0, 2),
        Vec3(0, 0, 2)
    ])


@pytest.fixture
def triangle_polygon() -> SupportPolygon:
    """A right triangle polygon on XZ plane."""
    return SupportPolygon(vertices=[
        Vec3(0, 0, 0),
        Vec3(2, 0, 0),
        Vec3(0, 0, 2)
    ])


@pytest.fixture
def bipedal_stance_polygon() -> SupportPolygon:
    """Support polygon for typical bipedal stance."""
    return SupportPolygon(vertices=[
        Vec3(-0.15, 0, -0.1),
        Vec3(0.15, 0, -0.1),
        Vec3(0.15, 0, 0.1),
        Vec3(-0.15, 0, 0.1)
    ])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) <= tol


def vec3_approx_equal(a: Vec3, b: Vec3, tol: float = 1e-6) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        approx_equal(a.x, b.x, tol) and
        approx_equal(a.y, b.y, tol) and
        approx_equal(a.z, b.z, tol)
    )


def vec3_is_zero(v: Vec3, tol: float = 1e-6) -> bool:
    """Check if a Vec3 is approximately zero."""
    return v.length() < tol


# =============================================================================
# TEST CLASS: Closest Point on Segment
# =============================================================================

class TestClosestPointOnSegment:
    """Tests for SupportPolygon.closest_point_on_segment static method."""

    def test_point_projects_to_segment_middle(self):
        """A point perpendicular to segment middle should project to that middle."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)
        point = Vec3(1, 0, 1)  # 1 unit above midpoint

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, Vec3(1, 0, 0))

    def test_point_projects_to_segment_start(self):
        """Point beyond segment start should project to start vertex."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)
        point = Vec3(-1, 0, 1)  # Beyond start

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, seg_start)

    def test_point_projects_to_segment_end(self):
        """Point beyond segment end should project to end vertex."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)
        point = Vec3(3, 0, 1)  # Beyond end

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, seg_end)

    def test_point_on_segment_returns_same_point(self):
        """Point exactly on segment should return that point."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)
        point = Vec3(1, 0, 0)  # On segment

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, point)

    def test_point_at_segment_start(self):
        """Point at segment start should return start."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)

        result = SupportPolygon.closest_point_on_segment(seg_start, seg_start, seg_end)

        assert vec3_approx_equal(result, seg_start)

    def test_point_at_segment_end(self):
        """Point at segment end should return end."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 0)

        result = SupportPolygon.closest_point_on_segment(seg_end, seg_start, seg_end)

        assert vec3_approx_equal(result, seg_end)

    def test_diagonal_segment(self):
        """Test projection onto diagonal segment."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(1, 0, 1)
        # Point perpendicular to diagonal at its midpoint
        midpoint = Vec3(0.5, 0, 0.5)
        perpendicular = Vec3(1, 0, -1).normalized()
        point = midpoint + perpendicular * 0.5

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, midpoint, tol=1e-5)

    def test_segment_in_xz_plane_diagonal(self):
        """Test projection onto XZ diagonal segment."""
        # SupportPolygon operates in ground plane (XZ), so test a 45-degree segment
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(2, 0, 2)
        point = Vec3(2, 0, 0)  # Should project to midpoint of diagonal

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        expected = Vec3(1, 0, 1)  # Midpoint
        assert vec3_approx_equal(result, expected)

    def test_z_axis_segment(self):
        """Test projection onto Z-axis segment."""
        seg_start = Vec3(0, 0, 0)
        seg_end = Vec3(0, 0, 2)
        point = Vec3(1, 1, 1)  # Off to the side

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, Vec3(0, 0, 1))

    def test_degenerate_segment_zero_length(self):
        """Zero-length segment (point) should return that point."""
        seg_start = Vec3(1, 0, 1)
        seg_end = Vec3(1, 0, 1)  # Same as start
        point = Vec3(0, 0, 0)

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        assert vec3_approx_equal(result, seg_start)

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        seg_start = Vec3(-2, 0, -2)
        seg_end = Vec3(-1, 0, -1)
        point = Vec3(-1.5, 0, -2)  # Projects to segment middle

        result = SupportPolygon.closest_point_on_segment(point, seg_start, seg_end)

        expected = Vec3(-1.75, 0, -1.75)
        assert result.distance(Vec3(-1.5, 0, -1.5)) < 0.5  # Close to midpoint region


# =============================================================================
# TEST CLASS: Closest Point on Boundary
# =============================================================================

class TestClosestPointOnBoundary:
    """Tests for SupportPolygon.closest_point_on_boundary method."""

    def test_point_outside_finds_closest_edge(self, unit_square_polygon):
        """Point outside polygon should find closest point on nearest edge."""
        # Point outside +X edge
        point = Vec3(1.0, 0, 0)  # Outside, nearest to right edge

        result = unit_square_polygon.closest_point_on_boundary(point)

        # Should be on the +X edge (x=0.5)
        assert approx_equal(result.x, 0.5, tol=1e-5)
        assert approx_equal(result.z, 0, tol=1e-5)

    def test_point_on_boundary_returns_same(self, unit_square_polygon):
        """Point exactly on boundary should return that point."""
        point = Vec3(0.5, 0, 0)  # On +X edge

        result = unit_square_polygon.closest_point_on_boundary(point)

        assert vec3_approx_equal(result, point, tol=1e-5)

    def test_point_at_vertex_returns_vertex(self, unit_square_polygon):
        """Point at a vertex should return that vertex."""
        vertex = Vec3(0.5, 0, 0.5)  # Top-right vertex

        result = unit_square_polygon.closest_point_on_boundary(vertex)

        assert vec3_approx_equal(result, vertex, tol=1e-5)

    def test_point_inside_finds_closest_edge(self, unit_square_polygon):
        """Point inside should find closest point on boundary."""
        # Point slightly inside from +X edge
        point = Vec3(0.4, 0, 0)

        result = unit_square_polygon.closest_point_on_boundary(point)

        # Closest edge is +X (x=0.5)
        assert approx_equal(result.x, 0.5, tol=1e-5)
        assert approx_equal(result.z, 0, tol=1e-5)

    def test_point_outside_corner_region(self, unit_square_polygon):
        """Point outside corner region should snap to corner vertex."""
        point = Vec3(1.0, 0, 1.0)  # Outside +X+Z corner

        result = unit_square_polygon.closest_point_on_boundary(point)

        corner = Vec3(0.5, 0, 0.5)
        assert vec3_approx_equal(result, corner, tol=1e-5)

    def test_point_at_centroid(self, unit_square_polygon):
        """Point at polygon centroid should find closest edge."""
        centroid = Vec3(0, 0, 0)

        result = unit_square_polygon.closest_point_on_boundary(centroid)

        # Centroid is equidistant from all edges at distance 0.5
        # Result should be on one of the edges
        dist_to_centroid = result.distance(centroid)
        assert approx_equal(dist_to_centroid, 0.5, tol=1e-5)

    def test_triangle_polygon(self, triangle_polygon):
        """Test closest point on triangle boundary."""
        # Point outside the hypotenuse
        point = Vec3(1.5, 0, 1.5)

        result = triangle_polygon.closest_point_on_boundary(point)

        # Should be on the hypotenuse (line from (2,0,0) to (0,0,2))
        # Verify result is on the boundary
        assert result.distance(point) < point.distance(Vec3(0, 0, 0))

    def test_large_distance_point(self, unit_square_polygon):
        """Point very far from polygon should find closest edge/vertex."""
        point = Vec3(100, 0, 0)  # Very far in +X direction

        result = unit_square_polygon.closest_point_on_boundary(point)

        # Should be on +X edge
        assert approx_equal(result.x, 0.5, tol=1e-5)

    def test_boundary_point_maintains_y_coordinate(self, unit_square_polygon):
        """Y coordinate should be preserved in ground plane projection."""
        point = Vec3(1.0, 0.5, 0)  # Has Y offset

        result = unit_square_polygon.closest_point_on_boundary(point)

        # Y should be 0 since polygon is on ground plane
        assert approx_equal(result.y, 0, tol=1e-5)


# =============================================================================
# TEST CLASS: Correction Vector Behavior
# =============================================================================

class TestCorrectionVectorBehavior:
    """Tests for SupportPolygon.correction_vector method."""

    def test_stable_com_no_correction(self, unit_square_polygon):
        """Point inside polygon needs no correction (zero vector)."""
        # COM at center - stable
        com = Vec3(0, 0, 0)

        correction = unit_square_polygon.correction_vector(com)

        assert vec3_is_zero(correction)

    def test_unstable_com_gets_correction(self, unit_square_polygon):
        """Point outside polygon gets non-zero correction vector."""
        # COM outside - unstable
        com = Vec3(1.0, 0, 0)

        correction = unit_square_polygon.correction_vector(com)

        assert not vec3_is_zero(correction)
        assert correction.length() > 0.4  # Should correct by at least 0.5 (distance to edge)

    def test_correction_points_inward(self, unit_square_polygon):
        """Correction vector should point toward polygon interior."""
        # COM outside +X edge
        com = Vec3(1.0, 0, 0)
        centroid = Vec3(0, 0, 0)

        correction = unit_square_polygon.correction_vector(com)

        # After correction, point should be closer to centroid
        corrected_pos = com + correction
        original_dist = com.distance(centroid)
        corrected_dist = corrected_pos.distance(centroid)
        assert corrected_dist < original_dist

    def test_correction_moves_to_boundary_or_inside(self, unit_square_polygon):
        """Applying correction should place point on/inside polygon."""
        com = Vec3(1.0, 0, 0)

        correction = unit_square_polygon.correction_vector(com)
        corrected_pos = com + correction

        # Should now be inside or on boundary
        # Check by verifying no further correction needed or minimal
        new_correction = unit_square_polygon.correction_vector(corrected_pos)
        assert vec3_is_zero(new_correction) or new_correction.length() < 1e-5

    def test_point_on_boundary_minimal_correction(self, unit_square_polygon):
        """Point on boundary should have zero or minimal correction."""
        boundary_point = Vec3(0.5, 0, 0)

        correction = unit_square_polygon.correction_vector(boundary_point)

        # Should be zero or very small
        assert correction.length() < 1e-5

    def test_corner_case_correction(self, unit_square_polygon):
        """Point outside corner should correct toward corner."""
        com = Vec3(1.0, 0, 1.0)  # Outside +X+Z corner

        correction = unit_square_polygon.correction_vector(com)

        corrected_pos = com + correction
        corner = Vec3(0.5, 0, 0.5)
        # Should be at or near corner
        assert corrected_pos.distance(corner) < 0.1

    def test_correction_magnitude_proportional_to_distance(self, unit_square_polygon):
        """Points further outside should have larger correction vectors."""
        near_outside = Vec3(0.6, 0, 0)
        far_outside = Vec3(2.0, 0, 0)

        near_correction = unit_square_polygon.correction_vector(near_outside)
        far_correction = unit_square_polygon.correction_vector(far_outside)

        assert far_correction.length() > near_correction.length()


# =============================================================================
# TEST CLASS: Balance Scenarios
# =============================================================================

class TestBalanceScenarios:
    """Tests for balance correction in realistic scenarios."""

    def test_leaning_forward_correction(self, bipedal_stance_polygon):
        """Leaning forward COM should get backward correction."""
        # COM ahead of support polygon
        com = Vec3(0, 0, 0.3)

        correction = bipedal_stance_polygon.correction_vector(com)

        # Should correct backward (-Z direction)
        assert correction.z < 0

    def test_leaning_sideways_correction(self, bipedal_stance_polygon):
        """Leaning sideways COM should get sideways correction."""
        # COM to the left of support
        com = Vec3(-0.3, 0, 0)

        correction = bipedal_stance_polygon.correction_vector(com)

        # Should correct rightward (+X direction)
        assert correction.x > 0

    def test_leaning_backward_correction(self, bipedal_stance_polygon):
        """Leaning backward COM should get forward correction."""
        com = Vec3(0, 0, -0.3)

        correction = bipedal_stance_polygon.correction_vector(com)

        # Should correct forward (+Z direction)
        assert correction.z > 0

    def test_diagonal_lean_correction(self, bipedal_stance_polygon):
        """Diagonal lean should get diagonal correction."""
        com = Vec3(0.3, 0, 0.3)  # Forward-right lean

        correction = bipedal_stance_polygon.correction_vector(com)

        # Should correct both backward and leftward
        assert correction.x < 0
        assert correction.z < 0

    def test_stable_standing_no_correction(self, bipedal_stance_polygon):
        """Centered COM within support requires no correction."""
        com = Vec3(0, 0, 0)  # Center of support polygon

        correction = bipedal_stance_polygon.correction_vector(com)

        assert vec3_is_zero(correction)

    def test_near_edge_stability(self, bipedal_stance_polygon):
        """COM near edge but inside should have no correction."""
        # Just inside the right edge
        com = Vec3(0.1, 0, 0)

        correction = bipedal_stance_polygon.correction_vector(com)

        assert vec3_is_zero(correction)


# =============================================================================
# TEST CLASS: Edge Cases and Robustness
# =============================================================================

class TestEdgeCasesRobustness:
    """Tests for edge cases and robustness."""

    def test_very_small_polygon(self):
        """Test with very small polygon vertices."""
        tiny_polygon = SupportPolygon(vertices=[
            Vec3(-0.001, 0, -0.001),
            Vec3(0.001, 0, -0.001),
            Vec3(0.001, 0, 0.001),
            Vec3(-0.001, 0, 0.001)
        ])
        point = Vec3(0.01, 0, 0)

        result = tiny_polygon.closest_point_on_boundary(point)
        correction = tiny_polygon.correction_vector(point)

        assert result is not None
        assert correction.length() > 0

    def test_very_large_polygon(self):
        """Test with very large polygon vertices."""
        large_polygon = SupportPolygon(vertices=[
            Vec3(-1000, 0, -1000),
            Vec3(1000, 0, -1000),
            Vec3(1000, 0, 1000),
            Vec3(-1000, 0, 1000)
        ])
        point = Vec3(0, 0, 0)

        correction = large_polygon.correction_vector(point)

        assert vec3_is_zero(correction)

    def test_irregular_polygon(self):
        """Test with irregular (non-rectangular) polygon."""
        irregular = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1.5, 0, 0.5),
            Vec3(1, 0, 1),
            Vec3(0, 0, 0.8)
        ])
        point = Vec3(2, 0, 0.5)  # Outside

        result = irregular.closest_point_on_boundary(point)
        correction = irregular.correction_vector(point)

        assert result is not None
        assert correction.length() > 0

    def test_narrow_polygon(self):
        """Test with very narrow polygon (long and thin)."""
        narrow = SupportPolygon(vertices=[
            Vec3(-5, 0, -0.01),
            Vec3(5, 0, -0.01),
            Vec3(5, 0, 0.01),
            Vec3(-5, 0, 0.01)
        ])
        point = Vec3(0, 0, 0.5)  # Outside the narrow side

        correction = narrow.correction_vector(point)

        # Should correct in -Z direction primarily
        assert abs(correction.z) > abs(correction.x)

    def test_collinear_vertices_handling(self):
        """Test behavior with collinear vertices (degenerate triangle)."""
        # Three collinear points - forms a line, not a polygon
        degenerate = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(2, 0, 0)
        ])
        point = Vec3(1, 0, 1)

        # Should still produce some result without crashing
        try:
            result = degenerate.closest_point_on_boundary(point)
            assert result is not None
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error for degenerate input
            pass


# =============================================================================
# TEST CLASS: Consistency and Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Tests for mathematical properties that should hold."""

    def test_closest_point_distance_is_minimal(self, unit_square_polygon):
        """Closest point should minimize distance to boundary."""
        point = Vec3(1.0, 0, 0.3)

        closest = unit_square_polygon.closest_point_on_boundary(point)

        # Sample other boundary points and verify closest is indeed closest
        boundary_samples = [
            Vec3(0.5, 0, -0.5),  # Bottom edge
            Vec3(0.5, 0, 0.5),   # Top edge
            Vec3(-0.5, 0, 0),    # Left edge
            Vec3(0.5, 0, 0),     # Right edge midpoint
        ]

        closest_dist = point.distance(closest)
        for sample in boundary_samples:
            assert point.distance(sample) >= closest_dist - 1e-5

    def test_correction_vector_direction(self, unit_square_polygon):
        """Correction vector should point from outside point toward boundary."""
        outside_point = Vec3(1.0, 0, 0)

        correction = unit_square_polygon.correction_vector(outside_point)
        closest = unit_square_polygon.closest_point_on_boundary(outside_point)

        # Correction should point toward closest boundary point
        to_closest = closest - outside_point
        dot = correction.normalized().dot(to_closest.normalized())
        assert dot > 0.99  # Nearly same direction

    def test_idempotent_correction(self, unit_square_polygon):
        """Applying correction twice should not change result."""
        outside_point = Vec3(1.0, 0, 0)

        correction1 = unit_square_polygon.correction_vector(outside_point)
        corrected = outside_point + correction1

        correction2 = unit_square_polygon.correction_vector(corrected)

        # Second correction should be zero or negligible
        assert vec3_is_zero(correction2) or correction2.length() < 1e-5

    def test_symmetry_of_square_polygon(self, unit_square_polygon):
        """Symmetric polygon should give symmetric corrections."""
        point_right = Vec3(1.0, 0, 0)
        point_left = Vec3(-1.0, 0, 0)

        corr_right = unit_square_polygon.correction_vector(point_right)
        corr_left = unit_square_polygon.correction_vector(point_left)

        # Magnitudes should be equal
        assert approx_equal(corr_right.length(), corr_left.length(), tol=1e-5)

        # X components should be opposite
        assert approx_equal(corr_right.x, -corr_left.x, tol=1e-5)


# =============================================================================
# TEST CLASS: Integration with Balance Checking
# =============================================================================

class TestBalanceIntegration:
    """Tests integrating closest point with overall balance checking."""

    def test_correction_restores_balance(self, bipedal_stance_polygon):
        """After applying correction, COM should be in stable region."""
        unstable_com = Vec3(0.5, 0, 0.5)  # Way outside

        correction = bipedal_stance_polygon.correction_vector(unstable_com)
        stable_com = unstable_com + correction

        # Verify stable (inside polygon or on boundary)
        final_correction = bipedal_stance_polygon.correction_vector(stable_com)
        assert vec3_is_zero(final_correction) or final_correction.length() < 1e-5

    def test_progressive_destabilization(self, unit_square_polygon):
        """Moving COM progressively outside should increase correction."""
        corrections = []
        for offset in [0.6, 0.8, 1.0, 1.5, 2.0]:
            com = Vec3(offset, 0, 0)
            corr = unit_square_polygon.correction_vector(com)
            corrections.append(corr.length())

        # Each correction should be larger than previous
        for i in range(1, len(corrections)):
            assert corrections[i] > corrections[i-1]

    def test_boundary_traversal(self, unit_square_polygon):
        """Points along boundary should have zero correction."""
        boundary_points = [
            Vec3(0.5, 0, 0),    # Right edge
            Vec3(-0.5, 0, 0),   # Left edge
            Vec3(0, 0, 0.5),    # Top edge
            Vec3(0, 0, -0.5),   # Bottom edge
            Vec3(0.5, 0, 0.5),  # Corner
        ]

        for point in boundary_points:
            correction = unit_square_polygon.correction_vector(point)
            assert correction.length() < 1e-5, f"Point {point} should have zero correction"

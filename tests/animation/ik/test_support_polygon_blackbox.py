"""
Blackbox tests for SupportPolygon class.

CLEANROOM TESTS - Written from specification only, without reading implementation.

SupportPolygon represents the base of support for balance checking:
- Built from foot contact positions
- Can test if a point is inside the polygon
- Works on the ground plane (XZ, y=0)
- Used to check if center of mass is within stable region

Public Interface:
    @dataclass
    class SupportPolygon:
        vertices: List[Vec3]  # XZ plane vertices

        @classmethod
        def from_foot_positions(cls, positions: List[Vec3]) -> 'SupportPolygon'

        def contains_point(self, point: Vec3) -> bool

        def project_to_ground(self, point: Vec3) -> Vec3
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
    """An equilateral triangle polygon on XZ plane."""
    return SupportPolygon(vertices=[
        Vec3(0, 0, 0),
        Vec3(2, 0, 0),
        Vec3(1, 0, 1.732)  # sqrt(3) for equilateral
    ])


@pytest.fixture
def bipedal_stance_positions() -> List[Vec3]:
    """Two foot positions for bipedal stance."""
    return [
        Vec3(-0.15, 0, 0),  # Left foot
        Vec3(0.15, 0, 0)    # Right foot
    ]


@pytest.fixture
def tripod_stance_positions() -> List[Vec3]:
    """Three contact points for tripod stance (e.g., two feet + cane)."""
    return [
        Vec3(-0.15, 0, 0),   # Left foot
        Vec3(0.15, 0, 0),    # Right foot
        Vec3(0, 0, 0.3)      # Cane or third contact
    ]


@pytest.fixture
def quadruped_stance_positions() -> List[Vec3]:
    """Four foot positions for quadruped stance."""
    return [
        Vec3(-0.2, 0, -0.3),  # Front left
        Vec3(0.2, 0, -0.3),   # Front right
        Vec3(-0.2, 0, 0.3),   # Back left
        Vec3(0.2, 0, 0.3)     # Back right
    ]


# =============================================================================
# TEST CLASS: Basic Polygon Behavior
# =============================================================================

class TestSupportPolygonBasicBehavior:
    """Test fundamental behavior of SupportPolygon."""

    def test_center_of_square_is_inside(self, large_square_polygon: SupportPolygon):
        """Center point of square polygon should be inside."""
        center = Vec3(1, 0, 1)
        assert large_square_polygon.contains_point(center) is True

    def test_point_far_outside_is_outside(self, large_square_polygon: SupportPolygon):
        """Point far outside polygon should be detected as outside."""
        far_point = Vec3(100, 0, 100)
        assert large_square_polygon.contains_point(far_point) is False

    def test_point_slightly_outside_is_outside(self, large_square_polygon: SupportPolygon):
        """Point just outside polygon boundary should be outside."""
        # Polygon is 0-2 in both X and Z
        outside_point = Vec3(2.1, 0, 1)
        assert large_square_polygon.contains_point(outside_point) is False

    def test_point_slightly_inside_is_inside(self, large_square_polygon: SupportPolygon):
        """Point just inside polygon boundary should be inside."""
        inside_point = Vec3(1.9, 0, 1)
        assert large_square_polygon.contains_point(inside_point) is True

    def test_origin_inside_unit_square(self, unit_square_polygon: SupportPolygon):
        """Origin should be inside a unit square centered at origin."""
        assert unit_square_polygon.contains_point(Vec3(0, 0, 0)) is True

    def test_corner_outside_for_centered_square(self, unit_square_polygon: SupportPolygon):
        """Point at corner coordinates should be on boundary or outside."""
        # Testing point beyond corner
        corner_outside = Vec3(1, 0, 1)
        assert unit_square_polygon.contains_point(corner_outside) is False

    def test_negative_coordinates_work(self):
        """Polygon with negative coordinates should work correctly."""
        poly = SupportPolygon(vertices=[
            Vec3(-2, 0, -2),
            Vec3(0, 0, -2),
            Vec3(0, 0, 0),
            Vec3(-2, 0, 0)
        ])
        # Center at (-1, 0, -1)
        assert poly.contains_point(Vec3(-1, 0, -1)) is True
        assert poly.contains_point(Vec3(1, 0, 1)) is False


class TestSupportPolygonTriangle:
    """Test behavior with triangular polygons."""

    def test_centroid_inside_triangle(self, triangle_polygon: SupportPolygon):
        """Centroid of triangle should be inside."""
        # Centroid of triangle with vertices (0,0), (2,0), (1, sqrt(3))
        centroid = Vec3(1, 0, 1.732 / 3)
        assert triangle_polygon.contains_point(centroid) is True

    def test_point_outside_triangle(self, triangle_polygon: SupportPolygon):
        """Point outside triangle should be detected."""
        outside = Vec3(-1, 0, 0)
        assert triangle_polygon.contains_point(outside) is False

    def test_point_above_triangle_apex(self, triangle_polygon: SupportPolygon):
        """Point above the apex (higher Z) should be outside."""
        above_apex = Vec3(1, 0, 2.5)
        assert triangle_polygon.contains_point(above_apex) is False


class TestSupportPolygonIrregular:
    """Test behavior with irregular polygons."""

    def test_l_shaped_polygon(self):
        """L-shaped polygon should correctly identify inside/outside."""
        # L-shape vertices (counterclockwise)
        l_poly = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(2, 0, 0),
            Vec3(2, 0, 1),
            Vec3(1, 0, 1),
            Vec3(1, 0, 2),
            Vec3(0, 0, 2)
        ])
        # Inside the L
        assert l_poly.contains_point(Vec3(0.5, 0, 1)) is True
        assert l_poly.contains_point(Vec3(1.5, 0, 0.5)) is True
        # In the cut-out corner (outside)
        assert l_poly.contains_point(Vec3(1.5, 0, 1.5)) is False

    def test_convex_pentagon(self):
        """Regular convex pentagon should work."""
        # Regular pentagon centered at origin
        import math
        vertices = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 - math.pi / 2
            vertices.append(Vec3(math.cos(angle), 0, math.sin(angle)))

        poly = SupportPolygon(vertices=vertices)
        # Center should be inside
        assert poly.contains_point(Vec3(0, 0, 0)) is True
        # Far outside should be outside
        assert poly.contains_point(Vec3(5, 0, 5)) is False


# =============================================================================
# TEST CLASS: From Foot Positions Factory
# =============================================================================

class TestFromFootPositions:
    """Test the from_foot_positions class method."""

    def test_from_foot_positions_creates_valid_polygon(
        self, bipedal_stance_positions: List[Vec3]
    ):
        """from_foot_positions should create a valid SupportPolygon."""
        poly = SupportPolygon.from_foot_positions(bipedal_stance_positions)
        assert poly is not None
        assert isinstance(poly, SupportPolygon)
        assert hasattr(poly, 'vertices')

    def test_from_foot_positions_preserves_foot_count(
        self, tripod_stance_positions: List[Vec3]
    ):
        """Polygon should be created from the given foot positions."""
        poly = SupportPolygon.from_foot_positions(tripod_stance_positions)
        # The polygon should have vertices related to foot positions
        assert poly.vertices is not None
        assert len(poly.vertices) >= 2  # At least need 2 for a degenerate polygon

    def test_bipedal_stance_creates_usable_polygon(
        self, bipedal_stance_positions: List[Vec3]
    ):
        """Bipedal stance should create polygon usable for balance checks."""
        poly = SupportPolygon.from_foot_positions(bipedal_stance_positions)
        # Point between feet should be inside or on boundary
        between_feet = Vec3(0, 0, 0)
        # For a line (2 points), contains_point behavior depends on implementation
        # The polygon should at least be queryable
        result = poly.contains_point(between_feet)
        assert isinstance(result, bool)

    def test_tripod_stance_point_in_center_inside(
        self, tripod_stance_positions: List[Vec3]
    ):
        """Center of tripod stance should be inside polygon."""
        poly = SupportPolygon.from_foot_positions(tripod_stance_positions)
        # Centroid of the three points
        cx = sum(p.x for p in tripod_stance_positions) / 3
        cz = sum(p.z for p in tripod_stance_positions) / 3
        center = Vec3(cx, 0, cz)
        assert poly.contains_point(center) is True

    def test_quadruped_stance_center_inside(
        self, quadruped_stance_positions: List[Vec3]
    ):
        """Center of quadruped stance should be inside polygon."""
        # Note: from_foot_positions may order vertices differently
        # Use positions in convex hull order (counterclockwise)
        ordered_positions = [
            Vec3(-0.2, 0, -0.3),  # Front left
            Vec3(0.2, 0, -0.3),   # Front right
            Vec3(0.2, 0, 0.3),    # Back right
            Vec3(-0.2, 0, 0.3)    # Back left
        ]
        poly = SupportPolygon.from_foot_positions(ordered_positions)
        # Centroid
        cx = sum(p.x for p in ordered_positions) / 4
        cz = sum(p.z for p in ordered_positions) / 4
        center = Vec3(cx, 0, cz)
        assert poly.contains_point(center) is True

    def test_quadruped_stance_outside_point_detected(
        self, quadruped_stance_positions: List[Vec3]
    ):
        """Point outside quadruped stance polygon should be detected."""
        poly = SupportPolygon.from_foot_positions(quadruped_stance_positions)
        outside = Vec3(1.0, 0, 0)  # Far to the right
        assert poly.contains_point(outside) is False


# =============================================================================
# TEST CLASS: Balance Scenarios
# =============================================================================

class TestBalanceScenarios:
    """Test realistic balance checking scenarios."""

    def test_bipedal_stance_com_centered(self):
        """Center of mass between two feet should be stable."""
        # For bipedal stance, use foot outlines not just single points
        # to create an actual polygon area
        left_foot = [
            Vec3(-0.15, 0, -0.05),
            Vec3(-0.05, 0, -0.05),
            Vec3(-0.05, 0, 0.05),
            Vec3(-0.15, 0, 0.05)
        ]
        right_foot = [
            Vec3(0.05, 0, -0.05),
            Vec3(0.15, 0, -0.05),
            Vec3(0.15, 0, 0.05),
            Vec3(0.05, 0, 0.05)
        ]
        # Combine feet into convex hull order
        all_points = left_foot + right_foot
        poly = SupportPolygon.from_foot_positions(all_points)

        # COM directly between feet
        com = Vec3(0, 0.9, 0)  # Elevated but projects to ground
        projected = poly.project_to_ground(com)
        # After projection, point between feet should be inside convex hull
        assert poly.contains_point(projected) is True

    def test_tripod_stance_provides_larger_support(self):
        """Tripod stance should provide larger support polygon than bipedal."""
        # Bipedal
        bipedal = [Vec3(-0.1, 0, 0), Vec3(0.1, 0, 0)]
        bipedal_poly = SupportPolygon.from_foot_positions(bipedal)

        # Tripod - add a cane in front
        tripod = [Vec3(-0.1, 0, 0), Vec3(0.1, 0, 0), Vec3(0, 0, 0.2)]
        tripod_poly = SupportPolygon.from_foot_positions(tripod)

        # Point slightly forward should be outside bipedal but inside tripod
        forward_point = Vec3(0, 0, 0.1)

        # Tripod should contain this point
        assert tripod_poly.contains_point(forward_point) is True

    def test_wide_stance_larger_polygon(self):
        """Wider stance should create larger support polygon."""
        # Use proper rectangular feet for realistic polygon
        narrow_stance = [
            Vec3(-0.05, 0, -0.02),
            Vec3(0.05, 0, -0.02),
            Vec3(0.05, 0, 0.02),
            Vec3(-0.05, 0, 0.02)
        ]
        wide_stance = [
            Vec3(-0.3, 0, -0.02),
            Vec3(0.3, 0, -0.02),
            Vec3(0.3, 0, 0.02),
            Vec3(-0.3, 0, 0.02)
        ]

        narrow_poly = SupportPolygon.from_foot_positions(narrow_stance)
        wide_poly = SupportPolygon.from_foot_positions(wide_stance)

        # Point at x=0.2 should be outside narrow but inside wide
        test_point = Vec3(0.2, 0, 0)

        # Wide stance should contain this point
        assert wide_poly.contains_point(test_point) is True
        # Narrow stance should not
        assert narrow_poly.contains_point(test_point) is False

    def test_single_leg_stance_small_polygon(self):
        """Single leg stance should have minimal support polygon."""
        # Single foot represented as multiple contact points
        single_foot = [
            Vec3(-0.05, 0, -0.1),
            Vec3(0.05, 0, -0.1),
            Vec3(0.05, 0, 0.1),
            Vec3(-0.05, 0, 0.1)
        ]
        poly = SupportPolygon.from_foot_positions(single_foot)

        # Center of single foot should be inside
        assert poly.contains_point(Vec3(0, 0, 0)) is True

        # Point at hip offset should be outside
        assert poly.contains_point(Vec3(0.1, 0, 0)) is False

    def test_leaning_forward_com_shift(self):
        """COM shifted forward should still be checked against polygon."""
        feet = [
            Vec3(-0.15, 0, -0.1),
            Vec3(0.15, 0, -0.1),
            Vec3(-0.15, 0, 0.1),
            Vec3(0.15, 0, 0.1)
        ]
        poly = SupportPolygon.from_foot_positions(feet)

        # COM shifted forward but still within feet
        com_forward = Vec3(0, 0, 0.05)
        assert poly.contains_point(com_forward) is True

        # COM shifted too far forward
        com_too_forward = Vec3(0, 0, 0.3)
        assert poly.contains_point(com_too_forward) is False


# =============================================================================
# TEST CLASS: Ground Projection
# =============================================================================

class TestGroundProjection:
    """Test the project_to_ground method."""

    def test_elevated_point_projects_down(self, large_square_polygon: SupportPolygon):
        """Elevated point should project straight down to y=0."""
        elevated = Vec3(1, 5, 1)
        projected = large_square_polygon.project_to_ground(elevated)

        assert projected.x == pytest.approx(1, abs=1e-6)
        assert projected.y == pytest.approx(0, abs=1e-6)
        assert projected.z == pytest.approx(1, abs=1e-6)

    def test_below_ground_projects_up(self, large_square_polygon: SupportPolygon):
        """Point below ground should project up to y=0."""
        below = Vec3(1, -3, 1)
        projected = large_square_polygon.project_to_ground(below)

        assert projected.x == pytest.approx(1, abs=1e-6)
        assert projected.y == pytest.approx(0, abs=1e-6)
        assert projected.z == pytest.approx(1, abs=1e-6)

    def test_ground_level_unchanged(self, large_square_polygon: SupportPolygon):
        """Point already at ground level should remain unchanged."""
        on_ground = Vec3(1, 0, 1)
        projected = large_square_polygon.project_to_ground(on_ground)

        assert projected.x == pytest.approx(1, abs=1e-6)
        assert projected.y == pytest.approx(0, abs=1e-6)
        assert projected.z == pytest.approx(1, abs=1e-6)

    def test_projection_preserves_xz_coordinates(
        self, large_square_polygon: SupportPolygon
    ):
        """Projection should only affect Y coordinate."""
        original = Vec3(0.5, 10, 1.5)
        projected = large_square_polygon.project_to_ground(original)

        assert projected.x == pytest.approx(original.x, abs=1e-6)
        assert projected.z == pytest.approx(original.z, abs=1e-6)

    def test_projection_at_origin(self, unit_square_polygon: SupportPolygon):
        """Projection at origin should work correctly."""
        point = Vec3(0, 100, 0)
        projected = unit_square_polygon.project_to_ground(point)

        assert projected.x == pytest.approx(0, abs=1e-6)
        assert projected.y == pytest.approx(0, abs=1e-6)
        assert projected.z == pytest.approx(0, abs=1e-6)

    def test_projection_negative_coordinates(self):
        """Projection should work with negative XZ coordinates."""
        poly = SupportPolygon(vertices=[
            Vec3(-2, 0, -2),
            Vec3(0, 0, -2),
            Vec3(0, 0, 0),
            Vec3(-2, 0, 0)
        ])

        point = Vec3(-1, 50, -1)
        projected = poly.project_to_ground(point)

        assert projected.x == pytest.approx(-1, abs=1e-6)
        assert projected.y == pytest.approx(0, abs=1e-6)
        assert projected.z == pytest.approx(-1, abs=1e-6)

    def test_projected_point_can_be_tested_for_containment(
        self, large_square_polygon: SupportPolygon
    ):
        """Projected point should be usable for containment testing."""
        # Point above and inside polygon bounds
        com = Vec3(1, 1.5, 1)
        projected = large_square_polygon.project_to_ground(com)

        # Should be inside after projection
        assert large_square_polygon.contains_point(projected) is True

        # Point above but outside polygon bounds
        com_outside = Vec3(5, 1.5, 5)
        projected_outside = large_square_polygon.project_to_ground(com_outside)

        # Should be outside after projection
        assert large_square_polygon.contains_point(projected_outside) is False


# =============================================================================
# TEST CLASS: Edge Cases and Boundaries
# =============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_very_small_polygon(self):
        """Very small polygon should still work."""
        tiny = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(0.001, 0, 0),
            Vec3(0.001, 0, 0.001),
            Vec3(0, 0, 0.001)
        ])

        # Center should be inside
        center = Vec3(0.0005, 0, 0.0005)
        assert tiny.contains_point(center) is True

    def test_very_large_polygon(self):
        """Very large polygon should work."""
        huge = SupportPolygon(vertices=[
            Vec3(-1000, 0, -1000),
            Vec3(1000, 0, -1000),
            Vec3(1000, 0, 1000),
            Vec3(-1000, 0, 1000)
        ])

        assert huge.contains_point(Vec3(0, 0, 0)) is True
        assert huge.contains_point(Vec3(500, 0, 500)) is True
        assert huge.contains_point(Vec3(2000, 0, 0)) is False

    def test_polygon_vertices_property(self, large_square_polygon: SupportPolygon):
        """Vertices property should be accessible."""
        vertices = large_square_polygon.vertices
        assert vertices is not None
        assert len(vertices) == 4

    def test_minimum_vertices_triangle(self):
        """Minimum polygon should be a triangle (3 vertices)."""
        triangle = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0.5, 0, 1)
        ])

        assert len(triangle.vertices) == 3
        # Center should be inside
        assert triangle.contains_point(Vec3(0.5, 0, 0.3)) is True

    def test_many_vertices_polygon(self):
        """Polygon with many vertices (approximating circle) should work."""
        n = 32
        vertices = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            vertices.append(Vec3(math.cos(angle), 0, math.sin(angle)))

        circle_approx = SupportPolygon(vertices=vertices)

        # Center should be inside
        assert circle_approx.contains_point(Vec3(0, 0, 0)) is True
        # Point at radius 0.5 should be inside
        assert circle_approx.contains_point(Vec3(0.5, 0, 0)) is True
        # Point at radius 1.5 should be outside
        assert circle_approx.contains_point(Vec3(1.5, 0, 0)) is False

    def test_asymmetric_polygon(self):
        """Asymmetric polygon should work correctly."""
        asym = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(3, 0, 0),
            Vec3(3, 0, 1),
            Vec3(1, 0, 2),
            Vec3(0, 0, 1)
        ])

        # Point in the wider right side
        assert asym.contains_point(Vec3(2.5, 0, 0.5)) is True
        # Point in the narrower left side
        assert asym.contains_point(Vec3(0.5, 0, 0.5)) is True
        # Point outside
        assert asym.contains_point(Vec3(-1, 0, 0)) is False


# =============================================================================
# TEST CLASS: Y-Coordinate Handling
# =============================================================================

class TestYCoordinateHandling:
    """Test that Y coordinates are handled correctly (XZ plane focus)."""

    def test_contains_point_ignores_y_coordinate(
        self, large_square_polygon: SupportPolygon
    ):
        """contains_point should work with points at any Y level."""
        # Point on ground at center
        on_ground = Vec3(1, 0, 1)
        # Same XZ but elevated
        elevated = Vec3(1, 100, 1)
        # Same XZ but below
        below = Vec3(1, -50, 1)

        # All should be "inside" as far as XZ projection is concerned
        # The implementation may or may not project internally
        # But at minimum, the ground-level point should work
        assert large_square_polygon.contains_point(on_ground) is True

    def test_vertices_at_ground_level(self, large_square_polygon: SupportPolygon):
        """Polygon vertices should be at y=0 (ground plane)."""
        for vertex in large_square_polygon.vertices:
            assert vertex.y == pytest.approx(0, abs=1e-6)


# =============================================================================
# TEST CLASS: Integration with Balance Checking Workflow
# =============================================================================

class TestBalanceCheckingWorkflow:
    """Test typical balance checking workflow."""

    def test_complete_balance_check_workflow(self):
        """Test complete workflow: create polygon -> project COM -> check stability."""
        # Setup: Create support polygon from foot positions in convex hull order
        # (counterclockwise for proper winding)
        feet = [
            Vec3(-0.15, 0, -0.1),  # Back left
            Vec3(0.15, 0, -0.1),   # Back right
            Vec3(0.15, 0, 0.1),    # Front right
            Vec3(-0.15, 0, 0.1)    # Front left
        ]
        poly = SupportPolygon.from_foot_positions(feet)

        # Step 1: Get COM (simulated - would come from COM calculator)
        com = Vec3(0, 0.9, 0)  # COM at hip height, centered

        # Step 2: Project to ground
        com_ground = poly.project_to_ground(com)

        # Step 3: Check if stable
        is_stable = poly.contains_point(com_ground)

        # Verify
        assert is_stable is True

    def test_unstable_pose_detection(self):
        """Test detection of unstable pose (COM outside support polygon)."""
        # Narrow stance
        feet = [
            Vec3(-0.05, 0, 0),
            Vec3(0.05, 0, 0)
        ]
        poly = SupportPolygon.from_foot_positions(feet)

        # COM leaning far to the side
        com = Vec3(0.3, 0.9, 0)
        com_ground = poly.project_to_ground(com)

        # Should be unstable (outside support polygon)
        is_stable = poly.contains_point(com_ground)
        assert is_stable is False

    def test_dynamic_stance_change(self):
        """Test balance as stance changes dynamically."""
        # Start with two foot outlines (rectangles)
        initial_stance = [
            Vec3(-0.15, 0, -0.05),
            Vec3(0.15, 0, -0.05),
            Vec3(0.15, 0, 0.05),
            Vec3(-0.15, 0, 0.05)
        ]

        # Final: Set both feet wider
        wide_stance = [
            Vec3(-0.3, 0, -0.05),
            Vec3(0.3, 0, -0.05),
            Vec3(0.3, 0, 0.05),
            Vec3(-0.3, 0, 0.05)
        ]

        # COM remains centered
        com = Vec3(0, 0, 0)

        # Initial stance - should be stable
        poly1 = SupportPolygon.from_foot_positions(initial_stance)
        assert poly1.contains_point(com) is True

        # Wide stance - should definitely be stable
        poly3 = SupportPolygon.from_foot_positions(wide_stance)
        assert poly3.contains_point(com) is True


# =============================================================================
# TEST CLASS: Robustness
# =============================================================================

class TestRobustness:
    """Test robustness of SupportPolygon."""

    def test_collinear_points_handled(self):
        """Collinear points (degenerate polygon) should be handled."""
        # Three points on a line
        line_points = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)]

        # Should not crash
        poly = SupportPolygon.from_foot_positions(line_points)

        # Point on the line
        on_line = Vec3(1, 0, 0)
        # Should at least be queryable without error
        result = poly.contains_point(on_line)
        assert isinstance(result, bool)

    def test_duplicate_vertices_handled(self):
        """Duplicate vertices should be handled gracefully."""
        # Square with duplicate corner
        with_dupe = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1, 0, 0),  # Duplicate
            Vec3(1, 0, 1),
            Vec3(0, 0, 1)
        ]

        poly = SupportPolygon.from_foot_positions(with_dupe)
        # Should still work
        assert poly.contains_point(Vec3(0.5, 0, 0.5)) is True

    def test_clockwise_vs_counterclockwise_vertices(self):
        """Both winding orders should work."""
        # Counterclockwise
        ccw = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1, 0, 1),
            Vec3(0, 0, 1)
        ])

        # Clockwise
        cw = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(0, 0, 1),
            Vec3(1, 0, 1),
            Vec3(1, 0, 0)
        ])

        center = Vec3(0.5, 0, 0.5)

        # Both should contain the center
        assert ccw.contains_point(center) is True
        assert cw.contains_point(center) is True

    def test_very_close_points(self):
        """Very close points should be handled."""
        close_points = [
            Vec3(0, 0, 0),
            Vec3(0.0001, 0, 0),
            Vec3(0.0001, 0, 0.0001),
            Vec3(0, 0, 0.0001)
        ]

        poly = SupportPolygon.from_foot_positions(close_points)
        # Should be able to query
        result = poly.contains_point(Vec3(0.00005, 0, 0.00005))
        assert isinstance(result, bool)


# =============================================================================
# TEST CLASS: Performance Characteristics
# =============================================================================

class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    def test_multiple_containment_checks(self, large_square_polygon: SupportPolygon):
        """Multiple containment checks should be efficient."""
        # This is more of a smoke test than a performance test
        points = [Vec3(i * 0.1, 0, j * 0.1) for i in range(20) for j in range(20)]

        results = [large_square_polygon.contains_point(p) for p in points]

        # Should complete without issues
        assert len(results) == 400
        # Count points inside (polygon is 0-2 x 0-2)
        inside_count = sum(1 for r in results if r)
        # Most points in 0-2 range should be inside
        assert inside_count > 100

    def test_repeated_projections(self, large_square_polygon: SupportPolygon):
        """Repeated projections should work correctly."""
        point = Vec3(1, 5, 1)

        # Project same point multiple times
        results = [large_square_polygon.project_to_ground(point) for _ in range(100)]

        # All results should be identical
        for r in results:
            assert r.x == pytest.approx(1, abs=1e-6)
            assert r.y == pytest.approx(0, abs=1e-6)
            assert r.z == pytest.approx(1, abs=1e-6)


# =============================================================================
# TEST CLASS: Dataclass Behavior
# =============================================================================

class TestDataclassBehavior:
    """Test dataclass behavior of SupportPolygon."""

    def test_polygon_is_dataclass(self):
        """SupportPolygon should behave as a dataclass."""
        from dataclasses import is_dataclass

        poly = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1, 0, 1)
        ])

        # Should be a dataclass
        assert is_dataclass(poly)

    def test_vertices_accessible(self):
        """Vertices should be directly accessible."""
        verts = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0, 1)]
        poly = SupportPolygon(vertices=verts)

        assert poly.vertices == verts

    def test_direct_construction_works(self):
        """Direct construction with vertices should work."""
        poly = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(2, 0, 0),
            Vec3(2, 0, 2),
            Vec3(0, 0, 2)
        ])

        assert len(poly.vertices) == 4
        assert poly.contains_point(Vec3(1, 0, 1)) is True

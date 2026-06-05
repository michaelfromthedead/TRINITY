"""
Tests for Voronoi Fracture System.

Whitebox tests for fracture_voronoi.py including:
- Vector math utilities
- Plane class and operations
- BoundingBox class
- Chunk validation and computations
- VoronoiCell operations
- Site distribution patterns
- Mesh clipping and fracturing
"""

import pytest
import math

from engine.simulation.destruction.fracture_voronoi import (
    # Type aliases
    Vec3,
    Triangle,
    # Vector utilities
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_dot,
    vec3_cross,
    vec3_length,
    vec3_normalize,
    vec3_distance,
    vec3_lerp,
    triangle_area,
    is_degenerate_triangle,
    # Classes
    Plane,
    BoundingBox,
    Chunk,
    VoronoiCell,
    SiteDistribution,
    VoronoiFracture,
    TetrahedralVoronoiFracture,
)
from engine.simulation.destruction.config import (
    MIN_CHUNK_VOLUME,
    MIN_VORONOI_SITES,
    MAX_VORONOI_SITES,
    DEGENERATE_TRIANGLE_AREA_THRESHOLD,
)


class TestVec3Add:
    """Tests for vec3_add function."""

    def test_add_zeros(self):
        """Adding zeros returns same vector."""
        result = vec3_add((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        assert result == (1.0, 2.0, 3.0)

    def test_add_positive(self):
        """Adding positive values."""
        result = vec3_add((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        assert result == (5.0, 7.0, 9.0)

    def test_add_negative(self):
        """Adding negative values."""
        result = vec3_add((1.0, 2.0, 3.0), (-1.0, -2.0, -3.0))
        assert result == (0.0, 0.0, 0.0)


class TestVec3Sub:
    """Tests for vec3_sub function."""

    def test_sub_same(self):
        """Subtracting same vector returns zero."""
        result = vec3_sub((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        assert result == (0.0, 0.0, 0.0)

    def test_sub_basic(self):
        """Basic subtraction."""
        result = vec3_sub((5.0, 7.0, 9.0), (1.0, 2.0, 3.0))
        assert result == (4.0, 5.0, 6.0)


class TestVec3Mul:
    """Tests for vec3_mul function."""

    def test_mul_by_one(self):
        """Multiplying by 1 returns same vector."""
        result = vec3_mul((1.0, 2.0, 3.0), 1.0)
        assert result == (1.0, 2.0, 3.0)

    def test_mul_by_zero(self):
        """Multiplying by 0 returns zero vector."""
        result = vec3_mul((1.0, 2.0, 3.0), 0.0)
        assert result == (0.0, 0.0, 0.0)

    def test_mul_by_scalar(self):
        """Multiplying by scalar."""
        result = vec3_mul((1.0, 2.0, 3.0), 2.0)
        assert result == (2.0, 4.0, 6.0)

    def test_mul_by_negative(self):
        """Multiplying by negative scalar."""
        result = vec3_mul((1.0, 2.0, 3.0), -1.0)
        assert result == (-1.0, -2.0, -3.0)


class TestVec3Dot:
    """Tests for vec3_dot function."""

    def test_dot_orthogonal(self):
        """Dot product of orthogonal vectors is zero."""
        result = vec3_dot((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert result == 0.0

    def test_dot_parallel(self):
        """Dot product of parallel vectors."""
        result = vec3_dot((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert result == 2.0

    def test_dot_antiparallel(self):
        """Dot product of antiparallel vectors is negative."""
        result = vec3_dot((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
        assert result == -1.0

    def test_dot_general(self):
        """General dot product."""
        result = vec3_dot((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        assert result == 32.0  # 1*4 + 2*5 + 3*6


class TestVec3Cross:
    """Tests for vec3_cross function."""

    def test_cross_x_y(self):
        """Cross product of X and Y is Z."""
        result = vec3_cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_cross_y_x(self):
        """Cross product of Y and X is -Z."""
        result = vec3_cross((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2] + 1.0) < 1e-10

    def test_cross_parallel_is_zero(self):
        """Cross product of parallel vectors is zero."""
        result = vec3_cross((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10


class TestVec3Length:
    """Tests for vec3_length function."""

    def test_length_zero(self):
        """Length of zero vector is zero."""
        assert vec3_length((0.0, 0.0, 0.0)) == 0.0

    def test_length_unit(self):
        """Length of unit vectors is 1."""
        assert abs(vec3_length((1.0, 0.0, 0.0)) - 1.0) < 1e-10
        assert abs(vec3_length((0.0, 1.0, 0.0)) - 1.0) < 1e-10
        assert abs(vec3_length((0.0, 0.0, 1.0)) - 1.0) < 1e-10

    def test_length_345(self):
        """Length of (3, 4, 0) is 5."""
        assert abs(vec3_length((3.0, 4.0, 0.0)) - 5.0) < 1e-10


class TestVec3Normalize:
    """Tests for vec3_normalize function."""

    def test_normalize_unit_vector(self):
        """Normalizing unit vector returns same."""
        result = vec3_normalize((1.0, 0.0, 0.0))
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_normalize_general(self):
        """Normalizing general vector."""
        result = vec3_normalize((3.0, 4.0, 0.0))
        assert abs(result[0] - 0.6) < 1e-10
        assert abs(result[1] - 0.8) < 1e-10
        length = vec3_length(result)
        assert abs(length - 1.0) < 1e-10

    def test_normalize_zero_vector(self):
        """Normalizing zero vector returns zero."""
        result = vec3_normalize((0.0, 0.0, 0.0))
        assert result == (0.0, 0.0, 0.0)

    def test_normalize_tiny_vector(self):
        """Normalizing tiny vector returns zero."""
        result = vec3_normalize((1e-12, 1e-12, 1e-12))
        assert result == (0.0, 0.0, 0.0)


class TestVec3Distance:
    """Tests for vec3_distance function."""

    def test_distance_same_point(self):
        """Distance from point to itself is zero."""
        assert vec3_distance((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == 0.0

    def test_distance_along_axis(self):
        """Distance along single axis."""
        assert abs(vec3_distance((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)) - 5.0) < 1e-10

    def test_distance_diagonal(self):
        """Distance in 3D diagonal."""
        # Distance from origin to (1, 1, 1) is sqrt(3)
        assert abs(vec3_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) - math.sqrt(3)) < 1e-10


class TestVec3Lerp:
    """Tests for vec3_lerp function."""

    def test_lerp_zero(self):
        """Lerp at t=0 returns first vector."""
        result = vec3_lerp((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 0.0)
        assert result == (0.0, 0.0, 0.0)

    def test_lerp_one(self):
        """Lerp at t=1 returns second vector."""
        result = vec3_lerp((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 1.0)
        assert result == (10.0, 10.0, 10.0)

    def test_lerp_half(self):
        """Lerp at t=0.5 returns midpoint."""
        result = vec3_lerp((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 0.5)
        assert result == (5.0, 5.0, 5.0)


class TestTriangleArea:
    """Tests for triangle_area function."""

    def test_unit_triangle_area(self):
        """Unit right triangle has area 0.5."""
        area = triangle_area(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        )
        assert abs(area - 0.5) < 1e-10

    def test_degenerate_triangle(self):
        """Collinear points have zero area."""
        area = triangle_area(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0)
        )
        assert abs(area) < 1e-10

    def test_scaled_triangle(self):
        """Scaled triangle has scaled area."""
        area = triangle_area(
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0)
        )
        assert abs(area - 2.0) < 1e-10


class TestIsDegenerateTriangle:
    """Tests for is_degenerate_triangle function."""

    def test_valid_triangle_not_degenerate(self):
        """Valid triangle is not degenerate."""
        result = is_degenerate_triangle(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        )
        assert result is False

    def test_collinear_points_degenerate(self):
        """Collinear points are degenerate."""
        result = is_degenerate_triangle(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0)
        )
        assert result is True

    def test_same_points_degenerate(self):
        """Same points are degenerate."""
        result = is_degenerate_triangle(
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0)
        )
        assert result is True

    def test_custom_threshold(self):
        """Custom threshold affects detection."""
        # Tiny triangle that would fail default threshold
        result = is_degenerate_triangle(
            (0.0, 0.0, 0.0),
            (1e-6, 0.0, 0.0),
            (0.0, 1e-6, 0.0),
            threshold=1e-15
        )
        assert result is False


class TestPlane:
    """Tests for Plane class."""

    def test_plane_construction(self):
        """Verify plane construction normalizes normal."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 2.0))
        assert abs(vec3_length(plane.normal) - 1.0) < 1e-10

    def test_signed_distance_on_plane(self):
        """Point on plane has zero distance."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        dist = plane.signed_distance((5.0, 5.0, 0.0))
        assert abs(dist) < 1e-10

    def test_signed_distance_in_front(self):
        """Point in front has positive distance."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        dist = plane.signed_distance((0.0, 0.0, 5.0))
        assert dist > 0

    def test_signed_distance_behind(self):
        """Point behind has negative distance."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        dist = plane.signed_distance((0.0, 0.0, -5.0))
        assert dist < 0

    def test_classify_point_front(self):
        """Classify point in front returns 1."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        assert plane.classify_point((0.0, 0.0, 5.0)) == 1

    def test_classify_point_back(self):
        """Classify point behind returns -1."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        assert plane.classify_point((0.0, 0.0, -5.0)) == -1

    def test_classify_point_on_plane(self):
        """Classify point on plane returns 0."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        assert plane.classify_point((5.0, 5.0, 0.0)) == 0

    def test_from_three_points(self):
        """Create plane from three points."""
        plane = Plane.from_three_points(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        )
        # Normal should be +Z or -Z
        assert abs(plane.normal[2]) > 0.99

    def test_bisector(self):
        """Create bisector plane between two points."""
        plane = Plane.bisector((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        # Plane should be at midpoint
        assert abs(plane.point[0] - 1.0) < 1e-10
        # Normal should be along X
        assert abs(plane.normal[0]) > 0.99


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        assert bbox.min_point == (0.0, 0.0, 0.0)
        assert bbox.max_point == (1.0, 1.0, 1.0)

    def test_center(self):
        """Verify center calculation."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 4.0, 6.0))
        center = bbox.center
        assert center == (1.0, 2.0, 3.0)

    def test_size(self):
        """Verify size calculation."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 4.0, 6.0))
        size = bbox.size
        assert size == (2.0, 4.0, 6.0)

    def test_volume(self):
        """Verify volume calculation."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 3.0, 4.0))
        assert bbox.volume == 24.0

    def test_contains_inside(self):
        """Point inside is contained."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        assert bbox.contains((0.5, 0.5, 0.5)) is True

    def test_contains_outside(self):
        """Point outside is not contained."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        assert bbox.contains((2.0, 0.5, 0.5)) is False

    def test_contains_on_boundary(self):
        """Point on boundary is contained."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        assert bbox.contains((1.0, 0.5, 0.5)) is True

    def test_expand(self):
        """Verify box expansion."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        expanded = bbox.expand(0.5)
        assert expanded.min_point == (-0.5, -0.5, -0.5)
        assert expanded.max_point == (1.5, 1.5, 1.5)

    def test_from_points(self):
        """Create bounding box from points."""
        points = [
            (0.0, 0.0, 0.0),
            (5.0, 3.0, 1.0),
            (2.0, 7.0, 4.0),
        ]
        bbox = BoundingBox.from_points(points)
        assert bbox.min_point == (0.0, 0.0, 0.0)
        assert bbox.max_point == (5.0, 7.0, 4.0)

    def test_from_empty_points(self):
        """Empty points list returns zero box."""
        bbox = BoundingBox.from_points([])
        assert bbox.min_point == (0, 0, 0)
        assert bbox.max_point == (0, 0, 0)


class TestChunk:
    """Tests for Chunk class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        triangles = [(0, 1, 2)]
        chunk = Chunk(vertices=vertices, triangles=triangles)
        assert len(chunk.vertices) == 3
        assert len(chunk.triangles) == 1

    def test_compute_centroid(self):
        """Verify centroid computation."""
        vertices = [
            (0.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 3.0)
        ]
        chunk = Chunk(vertices=vertices, triangles=[])
        centroid = chunk.compute_centroid()
        assert abs(centroid[0] - 0.75) < 1e-10
        assert abs(centroid[1] - 0.75) < 1e-10
        assert abs(centroid[2] - 0.75) < 1e-10

    def test_compute_centroid_empty(self):
        """Empty vertices returns zero centroid."""
        chunk = Chunk(vertices=[], triangles=[])
        centroid = chunk.compute_centroid()
        assert centroid == (0.0, 0.0, 0.0)

    def test_compute_volume_tetrahedron(self):
        """Verify volume computation for tetrahedron."""
        # Unit tetrahedron with vertices at origin and unit cube corners
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
        # Tetrahedron faces
        triangles = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]
        chunk = Chunk(vertices=vertices, triangles=triangles)
        volume = chunk.compute_volume()
        # Volume of this tetrahedron is 1/6
        assert abs(volume - 1.0/6.0) < 0.01

    def test_compute_volume_insufficient_vertices(self):
        """Insufficient vertices returns zero volume."""
        chunk = Chunk(vertices=[(0.0, 0.0, 0.0)], triangles=[])
        volume = chunk.compute_volume()
        assert volume == 0.0

    def test_get_bounds(self):
        """Verify bounds calculation."""
        vertices = [
            (0.0, 0.0, 0.0),
            (5.0, 3.0, 1.0),
            (2.0, 7.0, 4.0),
        ]
        chunk = Chunk(vertices=vertices, triangles=[])
        bounds = chunk.get_bounds()
        assert bounds.min_point == (0.0, 0.0, 0.0)
        assert bounds.max_point == (5.0, 7.0, 4.0)

    def test_is_valid(self):
        """Verify validity check."""
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
        triangles = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]
        chunk = Chunk(vertices=vertices, triangles=triangles)
        chunk.compute_volume()
        assert chunk.is_valid() is True

    def test_is_valid_tiny_chunk(self):
        """Tiny chunk is not valid."""
        vertices = [
            (0.0, 0.0, 0.0),
            (0.001, 0.0, 0.0),
            (0.0, 0.001, 0.0),
            (0.0, 0.0, 0.001)
        ]
        triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        chunk = Chunk(vertices=vertices, triangles=triangles)
        chunk.compute_volume()
        assert chunk.is_valid(min_volume=0.1) is False


class TestVoronoiCell:
    """Tests for VoronoiCell class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        cell = VoronoiCell(site=(1.0, 2.0, 3.0), index=5)
        assert cell.site == (1.0, 2.0, 3.0)
        assert cell.index == 5

    def test_contains_point_no_planes(self):
        """Cell with no planes contains all points."""
        cell = VoronoiCell(site=(0.0, 0.0, 0.0), index=0)
        assert cell.contains_point((10.0, 10.0, 10.0)) is True

    def test_contains_point_with_planes(self):
        """Cell with planes constrains containment."""
        cell = VoronoiCell(site=(0.0, 0.0, 0.0), index=0)
        # Add plane at x=1 with normal pointing +X
        cell.planes.append(Plane(point=(1.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)))

        # Point at x=0.5 should be inside (behind plane)
        assert cell.contains_point((0.5, 0.0, 0.0)) is True
        # Point at x=2 should be outside (in front of plane)
        assert cell.contains_point((2.0, 0.0, 0.0)) is False


class TestSiteDistribution:
    """Tests for SiteDistribution enumeration."""

    def test_all_distributions_exist(self):
        """Verify all distribution types exist."""
        assert hasattr(SiteDistribution, 'UNIFORM')
        assert hasattr(SiteDistribution, 'CLUSTERED')
        assert hasattr(SiteDistribution, 'SURFACE_BIASED')
        assert hasattr(SiteDistribution, 'IMPACT_CENTERED')


class TestVoronoiFracture:
    """Tests for VoronoiFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = VoronoiFracture(seed=42, num_sites=10)
        assert fracture.seed == 42
        assert fracture.num_sites == 10

    def test_num_sites_clamping(self):
        """Verify sites are clamped to valid range."""
        fracture_low = VoronoiFracture(num_sites=1)
        assert fracture_low.num_sites >= MIN_VORONOI_SITES

        fracture_high = VoronoiFracture(num_sites=1000)
        assert fracture_high.num_sites <= MAX_VORONOI_SITES

    def test_seed_setter(self):
        """Verify seed setter resets RNG."""
        fracture = VoronoiFracture(seed=42)
        fracture.seed = 99
        assert fracture.seed == 99

    def test_generate_uniform_sites(self):
        """Verify uniform site generation."""
        fracture = VoronoiFracture(seed=42, num_sites=10)
        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))

        sites = fracture.generate_voronoi_sites(bounds, SiteDistribution.UNIFORM)

        assert len(sites) == 10
        for site in sites:
            assert bounds.contains(site)

    def test_generate_clustered_sites(self):
        """Verify clustered site generation."""
        fracture = VoronoiFracture(seed=42, num_sites=20)
        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))

        sites = fracture.generate_voronoi_sites(bounds, SiteDistribution.CLUSTERED)

        assert len(sites) == 20
        for site in sites:
            assert bounds.contains(site)

    def test_generate_impact_centered_sites(self):
        """Verify impact-centered site generation."""
        fracture = VoronoiFracture(seed=42, num_sites=10)
        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))
        impact = (5.0, 5.0, 5.0)

        sites = fracture.generate_voronoi_sites(
            bounds,
            SiteDistribution.IMPACT_CENTERED,
            impact_point=impact
        )

        assert len(sites) == 10

    def test_compute_voronoi_cells(self):
        """Verify Voronoi cell computation."""
        fracture = VoronoiFracture(seed=42, num_sites=4)
        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))

        fracture.generate_voronoi_sites(bounds)
        cells = fracture.compute_voronoi_cells()

        assert len(cells) == 4
        for cell in cells:
            assert len(cell.planes) > 0
            # Each cell should have bisector planes with all other cells
            assert len(cell.neighbors) == 3  # 4-1 other cells

    def test_deterministic_generation(self):
        """Verify same seed produces same sites."""
        fracture1 = VoronoiFracture(seed=12345, num_sites=10)
        fracture2 = VoronoiFracture(seed=12345, num_sites=10)

        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))

        sites1 = fracture1.generate_voronoi_sites(bounds)
        sites2 = fracture2.generate_voronoi_sites(bounds)

        for s1, s2 in zip(sites1, sites2):
            assert abs(s1[0] - s2[0]) < 1e-10
            assert abs(s1[1] - s2[1]) < 1e-10
            assert abs(s1[2] - s2[2]) < 1e-10

    def test_fracture_simple_cube(self):
        """Verify fracturing a simple cube."""
        # Create a unit cube mesh
        vertices = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)
        ]
        triangles = [
            # Front
            (0, 1, 2), (0, 2, 3),
            # Back
            (4, 6, 5), (4, 7, 6),
            # Top
            (3, 2, 6), (3, 6, 7),
            # Bottom
            (0, 5, 1), (0, 4, 5),
            # Right
            (1, 5, 6), (1, 6, 2),
            # Left
            (0, 3, 7), (0, 7, 4)
        ]

        fracture = VoronoiFracture(seed=42, num_sites=4, min_chunk_volume=0.001)
        chunks = fracture.fracture(vertices, triangles)

        # Should produce some chunks
        assert len(chunks) > 0
        assert len(chunks) <= 4

        # All chunks should be valid
        for chunk in chunks:
            assert chunk.is_valid(min_volume=0.001)

    def test_sites_property(self):
        """Verify sites property returns copy."""
        fracture = VoronoiFracture(seed=42, num_sites=5)
        bounds = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))
        fracture.generate_voronoi_sites(bounds)

        sites = fracture.sites
        # Modifying returned list shouldn't affect internal state
        sites.append((0.0, 0.0, 0.0))
        assert len(fracture.sites) == 5


class TestTetrahedralVoronoiFracture:
    """Tests for TetrahedralVoronoiFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = TetrahedralVoronoiFracture(seed=42, num_sites=6)
        assert fracture.seed == 42

    def test_compute_tetrahedral_volume(self):
        """Verify tetrahedral volume computation."""
        fracture = TetrahedralVoronoiFracture()

        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
        tetrahedra = [(0, 1, 2, 3)]

        volume = fracture.compute_tetrahedral_volume(vertices, tetrahedra)
        # Volume of this tetrahedron is 1/6
        assert abs(volume - 1.0/6.0) < 0.01

    def test_fracture_tetrahedral_mesh(self):
        """Verify tetrahedral mesh fracturing."""
        fracture = TetrahedralVoronoiFracture(seed=42, num_sites=4)

        # Simple tetrahedron
        vertices = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 2.0, 0.0),
            (1.0, 1.0, 2.0)
        ]
        tetrahedra = [(0, 1, 2, 3)]

        chunks = fracture.fracture_tetrahedral(vertices, tetrahedra)
        # Should produce some chunks (may be empty if mesh too small)
        assert isinstance(chunks, list)

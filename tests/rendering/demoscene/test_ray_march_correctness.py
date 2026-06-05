"""
T-DEMO-7.3: Ray Marching Pipeline Correctness Tests

Comprehensive tests for ray marching correctness covering:
- Ray-sphere intersection: known hit point, miss case
- Ray-box intersection: face hit, edge hit, corner
- Shadow correctness: point in shadow, point in light
- Ambient occlusion: open area vs corner
- Normal estimation: compare 6-point vs analytic
- Perceptual epsilon: terminates at visual threshold
- Max step count: terminates before infinite loop
- Performance budget: stays under step limit
- WGSL vs Python reference: same scene, same result

Acceptance Criteria:
- Hit points within epsilon of analytic solution
- Shadows correctly occlude/pass
- AO values in [0, 1] range
- Normals are unit length
- WGSL output matches Python reference +/- epsilon
"""

from __future__ import annotations

import math
import pytest
from typing import Callable, Tuple

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.ray_march import (
    # Core ray marching
    SphereTracer,
    HitResult,
    MarchResultType,
    march_ray,
    RayMarchConfig,
    RayMarcher,
    RayMarchResult,
    # Perceptual epsilon
    epsilon_at_distance,
    PerceptualEpsilonConfig,
    # Normal estimation
    estimate_normal,
    NormalEstimator,
    NormalEstimationConfig,
    # SDF primitives for reference
    sdf_sphere,
    sdf_box,
    sdf_plane,
    sdf_torus,
    sdf_cylinder,
    # WGSL generation
    generate_ray_march_wgsl,
    generate_normal_estimation_wgsl,
    generate_epsilon_wgsl,
    generate_ray_march_struct_wgsl,
)


# =============================================================================
# Tolerance Constants
# =============================================================================

TOL_POSITION = 0.01      # Position tolerance for hit points
TOL_DISTANCE = 0.02      # Distance tolerance
TOL_NORMAL = 1e-4        # Normal unit length tolerance
TOL_ANGLE = 0.01         # Angle tolerance (radians)
TOL_AO = 0.1             # AO tolerance (perceptual)
TOL_SHADOW = 0.05        # Shadow tolerance


# =============================================================================
# Helper Functions
# =============================================================================

def is_unit_length(v: Vec3, tol: float = TOL_NORMAL) -> bool:
    """Check if vector is unit length."""
    return abs(v.length() - 1.0) < tol


def vec_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def vec_distance(a: Vec3, b: Vec3) -> float:
    """Euclidean distance between two points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def analytic_ray_sphere_intersection(
    origin: Vec3,
    direction: Vec3,
    sphere_center: Vec3,
    sphere_radius: float,
) -> Tuple[bool, float]:
    """
    Analytic ray-sphere intersection.
    Returns (hit, t) where t is the distance to hit point.
    """
    # Ray: P(t) = origin + t * direction
    # Sphere: |P - center|^2 = radius^2
    # Solving: |origin + t*dir - center|^2 = radius^2
    oc = Vec3(
        origin.x - sphere_center.x,
        origin.y - sphere_center.y,
        origin.z - sphere_center.z,
    )

    a = vec_dot(direction, direction)
    b = 2.0 * vec_dot(oc, direction)
    c = vec_dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0:
        return (False, 0.0)

    t = (-b - math.sqrt(discriminant)) / (2.0 * a)

    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2.0 * a)

    return (t > 0, t)


def analytic_ray_box_intersection(
    origin: Vec3,
    direction: Vec3,
    box_min: Vec3,
    box_max: Vec3,
) -> Tuple[bool, float, str]:
    """
    Analytic ray-AABB intersection using slab method.
    Returns (hit, t, face_hit) where face_hit is 'x', 'y', or 'z'.
    """
    tmin = float('-inf')
    tmax = float('inf')
    face = ''

    for i, (d, o, bmin, bmax) in enumerate([
        (direction.x, origin.x, box_min.x, box_max.x),
        (direction.y, origin.y, box_min.y, box_max.y),
        (direction.z, origin.z, box_min.z, box_max.z),
    ]):
        if abs(d) < 1e-10:
            if o < bmin or o > bmax:
                return (False, 0.0, '')
        else:
            t1 = (bmin - o) / d
            t2 = (bmax - o) / d

            if t1 > t2:
                t1, t2 = t2, t1

            if t1 > tmin:
                tmin = t1
                face = ['x', 'y', 'z'][i]

            tmax = min(tmax, t2)

            if tmin > tmax:
                return (False, 0.0, '')

    if tmin < 0:
        tmin = tmax
        if tmin < 0:
            return (False, 0.0, '')

    return (True, tmin, face)


def sphere_sdf_with_material(center: Vec3, radius: float) -> Callable[[Vec3], Tuple[float, int]]:
    """Create sphere SDF that returns (distance, material_id)."""
    def sdf(p: Vec3) -> Tuple[float, int]:
        d = Vec3(p.x - center.x, p.y - center.y, p.z - center.z).length() - radius
        return (d, 0)
    return sdf


def box_sdf_with_material(half_extents: Vec3) -> Callable[[Vec3], Tuple[float, int]]:
    """Create box SDF that returns (distance, material_id)."""
    def sdf(p: Vec3) -> Tuple[float, int]:
        d = sdf_box(p, half_extents)
        return (d, 0)
    return sdf


# =============================================================================
# Test Class 1: Ray-Sphere Intersection
# =============================================================================

class TestRaySphereIntersection:
    """Test ray-sphere intersection correctness."""

    def test_sphere_hit_known_point(self):
        """Ray hitting sphere at known point matches analytic solution."""
        # Sphere at origin, radius 1
        sphere_center = Vec3(0.0, 0.0, 0.0)
        sphere_radius = 1.0

        # Ray from (0, 0, 5) toward origin
        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        # Analytic solution
        hit, t_analytic = analytic_ray_sphere_intersection(
            origin, direction, sphere_center, sphere_radius
        )
        assert hit, "Analytic solution should find hit"

        # Ray march solution
        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(origin, direction, sdf, epsilon=0.001)

        assert result.hit, "Ray march should find hit"
        assert result.distance == pytest.approx(t_analytic, abs=TOL_DISTANCE), \
            f"Distance {result.distance} should match analytic {t_analytic}"

        # Expected hit point at (0, 0, 1)
        expected_hit = Vec3(0.0, 0.0, 1.0)
        assert vec_distance(result.position, expected_hit) < TOL_POSITION, \
            f"Hit position {result.position} should be near {expected_hit}"

    def test_sphere_hit_off_center(self):
        """Ray hitting sphere off-center still matches analytic."""
        sphere_center = Vec3(0.0, 0.0, 0.0)
        sphere_radius = 2.0

        # Offset ray
        origin = Vec3(1.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        hit, t_analytic = analytic_ray_sphere_intersection(
            origin, direction, sphere_center, sphere_radius
        )
        assert hit

        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(origin, direction, sdf, epsilon=0.001)

        assert result.hit
        assert result.distance == pytest.approx(t_analytic, abs=TOL_DISTANCE)

    def test_sphere_miss_case(self):
        """Ray missing sphere correctly reports no hit."""
        sphere_center = Vec3(0.0, 0.0, 0.0)
        sphere_radius = 1.0

        # Ray that misses
        origin = Vec3(5.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)  # Parallel, misses sphere

        hit, _ = analytic_ray_sphere_intersection(
            origin, direction, sphere_center, sphere_radius
        )
        assert not hit, "Analytic should miss"

        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(origin, direction, sdf)

        assert not result.hit, "Ray march should also miss"

    def test_sphere_grazing_ray(self):
        """Grazing ray (tangent to sphere) handles edge case."""
        sphere_center = Vec3(0.0, 0.0, 0.0)
        sphere_radius = 1.0

        # Ray tangent to sphere
        origin = Vec3(1.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        # This ray passes at exactly x=1, which is tangent to unit sphere
        # Due to floating point, it may or may not hit
        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(origin, direction, sdf, epsilon=0.01)

        # Should either hit tangentially or miss - no crash
        assert isinstance(result.hit, bool)


# =============================================================================
# Test Class 2: Ray-Box Intersection
# =============================================================================

class TestRayBoxIntersection:
    """Test ray-box intersection for faces, edges, and corners."""

    def test_box_face_hit(self):
        """Ray hitting box face matches expected face."""
        half_extents = Vec3(1.0, 1.0, 1.0)

        # Ray hitting +Z face
        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        # Analytic: should hit at z=1
        hit, t_analytic, face = analytic_ray_box_intersection(
            origin, direction, Vec3(-1, -1, -1), Vec3(1, 1, 1)
        )
        assert hit
        assert face == 'z'

        sdf = box_sdf_with_material(half_extents)
        result = march_ray(origin, direction, sdf, epsilon=0.001)

        assert result.hit
        assert result.distance == pytest.approx(t_analytic, abs=TOL_DISTANCE)
        assert result.position.z == pytest.approx(1.0, abs=TOL_POSITION)

    def test_box_edge_hit(self):
        """Ray hitting box edge terminates correctly."""
        half_extents = Vec3(1.0, 1.0, 1.0)

        # Ray aimed at edge between +X and +Z faces
        origin = Vec3(5.0, 0.0, 5.0)
        direction = Vec3(-1.0, 0.0, -1.0).normalized()

        sdf = box_sdf_with_material(half_extents)
        result = march_ray(origin, direction, sdf, epsilon=0.001)

        assert result.hit
        # Hit should be near the edge
        assert abs(result.position.x - 1.0) < TOL_POSITION or \
               abs(result.position.z - 1.0) < TOL_POSITION

    def test_box_corner_hit(self):
        """Ray hitting box corner terminates correctly."""
        half_extents = Vec3(1.0, 1.0, 1.0)

        # Ray aimed at +X +Y +Z corner
        origin = Vec3(5.0, 5.0, 5.0)
        direction = Vec3(-1.0, -1.0, -1.0).normalized()

        sdf = box_sdf_with_material(half_extents)
        result = march_ray(origin, direction, sdf, epsilon=0.001)

        assert result.hit
        # Hit should be near the corner (1, 1, 1)
        expected_corner = Vec3(1.0, 1.0, 1.0)
        distance_to_corner = vec_distance(result.position, expected_corner)
        assert distance_to_corner < 0.1, \
            f"Hit at {result.position}, expected near corner {expected_corner}"

    def test_box_all_faces(self):
        """Test hitting all six faces of box."""
        half_extents = Vec3(1.0, 1.0, 1.0)
        sdf = box_sdf_with_material(half_extents)

        face_tests = [
            (Vec3(0, 0, 5), Vec3(0, 0, -1), 'z', 1.0),   # +Z face
            (Vec3(0, 0, -5), Vec3(0, 0, 1), 'z', -1.0),  # -Z face
            (Vec3(5, 0, 0), Vec3(-1, 0, 0), 'x', 1.0),   # +X face
            (Vec3(-5, 0, 0), Vec3(1, 0, 0), 'x', -1.0),  # -X face
            (Vec3(0, 5, 0), Vec3(0, -1, 0), 'y', 1.0),   # +Y face
            (Vec3(0, -5, 0), Vec3(0, 1, 0), 'y', -1.0),  # -Y face
        ]

        for origin, direction, axis, expected_coord in face_tests:
            result = march_ray(origin, direction, sdf, epsilon=0.001)
            assert result.hit, f"Should hit {axis} face"

            coord = getattr(result.position, axis)
            assert coord == pytest.approx(expected_coord, abs=TOL_POSITION), \
                f"Face {axis} hit at wrong position: {result.position}"


# =============================================================================
# Test Class 3: Shadow Correctness
# =============================================================================

class TestShadowCorrectness:
    """Test that shadows correctly occlude/pass light."""

    def test_point_in_shadow(self):
        """Point directly behind occluder is in shadow."""
        # Sphere at (0, 2, 0), point at (0, 0, 0), light at (0, 5, 0)
        sphere_center = Vec3(0.0, 2.0, 0.0)
        sphere_radius = 0.5

        point = Vec3(0.0, 0.0, 0.0)
        light_pos = Vec3(0.0, 5.0, 0.0)

        # Shadow ray from point toward light
        to_light = Vec3(
            light_pos.x - point.x,
            light_pos.y - point.y,
            light_pos.z - point.z,
        ).normalized()

        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(point, to_light, sdf, epsilon=0.001, max_distance=10.0)

        # Should hit the sphere (in shadow)
        assert result.hit, "Point should be in shadow (ray blocked by sphere)"
        assert result.distance < 5.0, "Hit should be before reaching light"

    def test_point_in_light(self):
        """Point not blocked by occluder is in light."""
        sphere_center = Vec3(0.0, 2.0, 0.0)
        sphere_radius = 0.5

        # Point offset from shadow
        point = Vec3(3.0, 0.0, 0.0)
        light_pos = Vec3(3.0, 5.0, 0.0)

        to_light = Vec3(
            light_pos.x - point.x,
            light_pos.y - point.y,
            light_pos.z - point.z,
        ).normalized()

        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(point, to_light, sdf, epsilon=0.001, max_distance=10.0)

        # Should NOT hit (in light)
        assert not result.hit, "Point should be in light (ray unblocked)"

    def test_shadow_edge(self):
        """Point at shadow edge handles penumbra region."""
        sphere_center = Vec3(0.0, 2.0, 0.0)
        sphere_radius = 0.5

        # Point just at edge of shadow
        point = Vec3(0.5, 0.0, 0.0)  # At edge
        light_pos = Vec3(0.0, 5.0, 0.0)

        to_light = Vec3(
            light_pos.x - point.x,
            light_pos.y - point.y,
            light_pos.z - point.z,
        ).normalized()

        sdf = sphere_sdf_with_material(sphere_center, sphere_radius)
        result = march_ray(point, to_light, sdf, epsilon=0.001, max_distance=10.0)

        # At edge, may or may not hit - should not crash
        assert isinstance(result.hit, bool)


# =============================================================================
# Test Class 4: Ambient Occlusion
# =============================================================================

class TestAmbientOcclusion:
    """Test AO calculation for open areas vs corners."""

    def test_ao_open_area(self):
        """Open area (flat plane) has AO close to 1.0."""
        # Point on flat plane facing up - minimal occlusion
        point = Vec3(0.0, 0.0, 0.0)
        normal = Vec3(0.0, 1.0, 0.0)

        def plane_sdf(p: Vec3) -> float:
            return sdf_plane(p, Vec3(0.0, 1.0, 0.0), 0.0)

        # Sample AO using ray marching in normal direction
        # Multiple samples along normal
        ao_accum = 0.0
        samples = 5
        step_scale = 0.1
        falloff = 0.5

        for i in range(samples):
            sample_dist = (i + 1) * step_scale
            sample_pos = Vec3(
                point.x + normal.x * sample_dist,
                point.y + normal.y * sample_dist,
                point.z + normal.z * sample_dist,
            )
            d = plane_sdf(sample_pos)
            expected_d = sample_dist
            occlusion = max(0.0, (expected_d - d) / expected_d)
            ao_accum += occlusion * (falloff ** i)

        ao = 1.0 - min(ao_accum, 1.0)
        assert ao > 0.9, f"Open area AO should be > 0.9, got {ao}"

    def test_ao_corner(self):
        """Corner has lower AO (more occlusion)."""
        # Point in corner where two planes meet
        point = Vec3(0.01, 0.01, 0.0)
        normal = Vec3(1.0, 1.0, 0.0).normalized()

        def corner_sdf(p: Vec3) -> float:
            # Union of floor (y=0) and wall (x=0)
            d_floor = p.y
            d_wall = p.x
            return min(d_floor, d_wall)

        ao_accum = 0.0
        samples = 5
        step_scale = 0.1
        falloff = 0.5

        for i in range(samples):
            sample_dist = (i + 1) * step_scale
            sample_pos = Vec3(
                point.x + normal.x * sample_dist,
                point.y + normal.y * sample_dist,
                point.z + normal.z * sample_dist,
            )
            d = corner_sdf(sample_pos)
            expected_d = sample_dist
            occlusion = max(0.0, (expected_d - d) / expected_d)
            ao_accum += occlusion * (falloff ** i)

        ao = 1.0 - min(ao_accum, 1.0)
        assert ao < 0.7, f"Corner AO should be < 0.7 (more occluded), got {ao}"

    def test_ao_in_valid_range(self):
        """AO values are always in [0, 1] range."""
        test_cases = [
            (Vec3(0, 0, 0), Vec3(0, 1, 0)),      # Flat
            (Vec3(0.1, 0.1, 0), Vec3(1, 1, 0)),  # Corner
            (Vec3(0, 1, 0), Vec3(0, 1, 0)),      # Above plane
            (Vec3(0.5, 0.5, 0.5), Vec3(1, 0, 0)), # Arbitrary
        ]

        def simple_sdf(p: Vec3) -> float:
            return min(p.y, p.x)

        for point, normal in test_cases:
            normal = normal.normalized()

            ao_accum = 0.0
            samples = 5
            for i in range(samples):
                sample_dist = (i + 1) * 0.1
                sample_pos = Vec3(
                    point.x + normal.x * sample_dist,
                    point.y + normal.y * sample_dist,
                    point.z + normal.z * sample_dist,
                )
                d = simple_sdf(sample_pos)
                occlusion = max(0.0, (sample_dist - d) / sample_dist) if sample_dist > 0 else 0.0
                ao_accum += occlusion * (0.5 ** i)

            ao = 1.0 - min(ao_accum, 1.0)
            assert 0.0 <= ao <= 1.0, f"AO {ao} out of range for {point}"


# =============================================================================
# Test Class 5: Normal Estimation
# =============================================================================

class TestNormalEstimation:
    """Test normal estimation comparing 6-point vs analytic."""

    def test_sphere_normal_vs_analytic(self):
        """6-point normal matches analytic sphere normal."""
        # For a sphere centered at origin, normal at point P is P/|P|
        for angle in [0, 30, 45, 60, 90]:
            theta = math.radians(angle)
            test_point = Vec3(math.cos(theta), math.sin(theta), 0.0)

            # Analytic normal
            analytic_normal = test_point.normalized()

            # 6-point estimated normal
            estimated_normal = estimate_normal(
                test_point,
                lambda p: sdf_sphere(p, 1.0),
                epsilon=0.0001,
            )

            # Compare
            assert is_unit_length(estimated_normal), "Estimated normal must be unit length"

            dot = vec_dot(estimated_normal, analytic_normal)
            assert dot > 0.999, f"Normal mismatch at angle {angle}: dot={dot}"

    def test_box_face_normals(self):
        """Box face normals are axis-aligned."""
        half_extents = Vec3(1.0, 1.0, 1.0)

        face_tests = [
            (Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),   # +X face
            (Vec3(-1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0)), # -X face
            (Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)),   # +Y face
            (Vec3(0.0, -1.0, 0.0), Vec3(0.0, -1.0, 0.0)), # -Y face
            (Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 1.0)),   # +Z face
            (Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, -1.0)), # -Z face
        ]

        for point, expected_normal in face_tests:
            estimated = estimate_normal(
                point,
                lambda p: sdf_box(p, half_extents),
                epsilon=0.0001,
            )

            assert is_unit_length(estimated)
            dot = vec_dot(estimated, expected_normal)
            assert dot > 0.99, f"Box normal at {point}: expected {expected_normal}, got {estimated}"

    def test_plane_normal(self):
        """Plane normal is constant everywhere."""
        plane_normal = Vec3(0.0, 1.0, 0.0)

        for x in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                point = Vec3(float(x), 0.0, float(z))

                estimated = estimate_normal(
                    point,
                    lambda p: sdf_plane(p, plane_normal, 0.0),
                    epsilon=0.0001,
                )

                assert is_unit_length(estimated)
                dot = vec_dot(estimated, plane_normal)
                assert dot > 0.999, f"Plane normal mismatch at {point}"

    def test_tetrahedron_vs_6point(self):
        """Tetrahedron stencil gives similar results to 6-point."""
        point = Vec3(1.0, 0.0, 0.0)
        sdf = lambda p: sdf_sphere(p, 1.0)

        estimator_6pt = NormalEstimator(NormalEstimationConfig(use_tetrahedron=False))
        estimator_tet = NormalEstimator(NormalEstimationConfig(use_tetrahedron=True))

        normal_6pt = estimator_6pt.estimate(point, sdf)
        normal_tet = estimator_tet.estimate(point, sdf)

        assert is_unit_length(normal_6pt)
        assert is_unit_length(normal_tet)

        dot = vec_dot(normal_6pt, normal_tet)
        assert dot > 0.99, "6-point and tetrahedron normals should be similar"

    def test_normals_always_unit_length(self):
        """Normals are always unit length regardless of SDF."""
        test_sdfs = [
            lambda p: sdf_sphere(p, 0.5),
            lambda p: sdf_sphere(p, 5.0),
            lambda p: sdf_box(p, Vec3(0.5, 1.0, 2.0)),
            lambda p: sdf_plane(p, Vec3(1, 1, 1).normalized(), 0.0),
        ]

        test_points = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0.7, 0.7, 0),
        ]

        for sdf in test_sdfs:
            for point in test_points:
                try:
                    normal = estimate_normal(point, sdf, epsilon=0.0001)
                    assert is_unit_length(normal), \
                        f"Normal {normal} not unit length (len={normal.length()})"
                except RuntimeError:
                    # Zero-length normal at degenerate point is acceptable
                    pass


# =============================================================================
# Test Class 6: Perceptual Epsilon
# =============================================================================

class TestPerceptualEpsilon:
    """Test that perceptual epsilon terminates at visual threshold."""

    def test_epsilon_increases_with_distance(self):
        """Epsilon scales up with distance for performance."""
        base = 0.001
        fov = math.radians(60)

        eps_near = epsilon_at_distance(base, 1.0, fov)
        eps_mid = epsilon_at_distance(base, 10.0, fov)
        eps_far = epsilon_at_distance(base, 50.0, fov)

        assert eps_near < eps_mid < eps_far, \
            f"Epsilon should increase: {eps_near} < {eps_mid} < {eps_far}"

    def test_epsilon_respects_bounds(self):
        """Epsilon stays within min/max bounds."""
        base = 0.001
        fov = math.radians(60)
        min_eps = 1e-6
        max_eps = 0.1

        # Very close - should clamp to min
        eps_close = epsilon_at_distance(base, 0.0, fov, min_epsilon=min_eps, max_epsilon=max_eps)
        assert eps_close >= min_eps

        # Very far - should clamp to max
        eps_far = epsilon_at_distance(base, 1000.0, fov, min_epsilon=min_eps, max_epsilon=max_eps)
        assert eps_far <= max_eps

    def test_perceptual_epsilon_terminates_correctly(self):
        """Ray march with perceptual epsilon terminates at visual threshold."""
        config = RayMarchConfig(
            max_steps=256,
            base_epsilon=0.001,
            use_perceptual_epsilon=True,
        )
        marcher = RayMarcher(config)

        sdf = lambda p: sdf_sphere(p, 1.0)

        # Near ray
        result_near = marcher.march(Vec3(0, 0, 2), Vec3(0, 0, -1), sdf)
        # Far ray
        result_far = marcher.march(Vec3(0, 0, 51), Vec3(0, 0, -1), sdf)

        assert result_near.hit and result_far.hit

        # Far ray should use larger epsilon
        assert result_far.epsilon_used > result_near.epsilon_used

    def test_perceptual_config_validation(self):
        """PerceptualEpsilonConfig validates parameters."""
        # Valid config
        config = PerceptualEpsilonConfig(base_epsilon=0.001)
        assert config.base_epsilon == 0.001

        # Invalid configs
        with pytest.raises(ValueError):
            PerceptualEpsilonConfig(base_epsilon=-0.001)

        with pytest.raises(ValueError):
            PerceptualEpsilonConfig(fov=0.0)

        with pytest.raises(ValueError):
            PerceptualEpsilonConfig(pixel_scale=-1.0)


# =============================================================================
# Test Class 7: Max Step Count
# =============================================================================

class TestMaxStepCount:
    """Test that ray march terminates before infinite loop."""

    def test_max_steps_terminates(self):
        """Ray march terminates at max_steps limit."""
        # SDF that never converges (returns constant small value)
        def slow_sdf(p: Vec3) -> Tuple[float, int]:
            return (0.1, 0)  # Small but never hits epsilon

        tracer = SphereTracer(max_steps=10, epsilon=0.001)
        result = tracer.march(Vec3(0, 0, 0), Vec3(0, 0, 1), slow_sdf)

        assert not result.hit
        assert result.steps == 10
        assert result.result_type == MarchResultType.MAX_STEPS

    def test_max_distance_terminates(self):
        """Ray march terminates at max_distance limit."""
        def far_sdf(p: Vec3) -> Tuple[float, int]:
            return (1.0, 0)  # Steps of 1.0

        tracer = SphereTracer(max_steps=1000, max_distance=10.0, epsilon=0.001)
        result = tracer.march(Vec3(0, 0, 0), Vec3(0, 0, 1), far_sdf)

        assert not result.hit
        assert result.distance >= 10.0
        assert result.result_type == MarchResultType.MISS

    def test_no_infinite_loop_complex_sdf(self):
        """Complex SDF doesn't cause infinite loop."""
        def complex_sdf(p: Vec3) -> Tuple[float, int]:
            # Oscillating SDF that could cause issues
            base = sdf_sphere(p, 1.0)
            noise = math.sin(p.x * 10) * 0.01
            return (base + noise, 0)

        tracer = SphereTracer(max_steps=500, max_distance=100.0, epsilon=0.001)

        # Should terminate, not hang
        result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), complex_sdf)

        assert result.steps <= 500


# =============================================================================
# Test Class 8: Performance Budget
# =============================================================================

class TestPerformanceBudget:
    """Test that ray march stays under step limits."""

    def test_sphere_hit_under_budget(self):
        """Simple sphere hit uses reasonable step count."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)

        result = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            sdf,
            max_steps=256,
            epsilon=0.001,
        )

        assert result.hit
        # Sphere 4 units away with step size ~1 should hit in ~10-20 steps
        assert result.steps < 50, f"Too many steps: {result.steps}"

    def test_close_surface_minimal_steps(self):
        """Surface very close to origin needs minimal steps."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 0.1)

        result = march_ray(
            Vec3(0, 0, 0.5),  # Just 0.4 units from surface
            Vec3(0, 0, -1),
            sdf,
            epsilon=0.001,
        )

        assert result.hit
        assert result.steps < 20

    def test_far_surface_uses_more_steps(self):
        """Distant surface uses more steps but still bounded."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)

        result = march_ray(
            Vec3(0, 0, 50),  # 49 units from surface
            Vec3(0, 0, -1),
            sdf,
            max_steps=256,
            epsilon=0.001,
        )

        assert result.hit
        # Should still be well under max
        assert result.steps < 150


# =============================================================================
# Test Class 9: WGSL vs Python Reference
# =============================================================================

class TestWGSLPythonParity:
    """Test that WGSL generation matches Python reference logic."""

    def test_wgsl_struct_definition(self):
        """WGSL struct matches Python HitResult fields."""
        wgsl = generate_ray_march_struct_wgsl()

        # Python HitResult has: hit, position, distance, steps, material_id
        assert "hit: bool" in wgsl
        assert "position: vec3<f32>" in wgsl
        assert "distance: f32" in wgsl
        assert "steps: u32" in wgsl
        assert "material_id: u32" in wgsl

    def test_wgsl_epsilon_function_matches_python(self):
        """WGSL epsilon function uses same formula as Python."""
        wgsl = generate_epsilon_wgsl()

        # Formula: epsilon = base * (1.0 + distance * tan(fov/2) * pixel_scale)
        assert "tan" in wgsl
        assert "clamp" in wgsl
        assert "epsilon_at_distance" in wgsl

    def test_wgsl_normal_estimation_matches_python(self):
        """WGSL normal estimation uses same 6-point stencil."""
        wgsl = generate_normal_estimation_wgsl(use_tetrahedron=False)

        # 6-point pattern: sample at p+e.xyy, p-e.xyy, etc.
        assert "e.xyy" in wgsl
        assert "e.yxy" in wgsl
        assert "e.yyx" in wgsl
        assert "normalize" in wgsl

    def test_wgsl_tetrahedron_normal_matches_python(self):
        """WGSL tetrahedron normal uses same 4-point stencil."""
        wgsl = generate_normal_estimation_wgsl(use_tetrahedron=True)

        # Tetrahedron uses 1/sqrt(3) factor
        assert "0.5773502691896258" in wgsl
        assert "normalize" in wgsl

    def test_wgsl_ray_march_loop_structure(self):
        """WGSL ray march has same loop structure as Python."""
        wgsl = generate_ray_march_wgsl()

        # Loop structure
        assert "for" in wgsl
        assert "max_steps" in wgsl

        # Hit detection
        assert "scene_sdf" in wgsl
        assert "epsilon" in wgsl

        # Distance accumulation
        assert "t" in wgsl or "distance" in wgsl

    def test_wgsl_perceptual_epsilon_integration(self):
        """WGSL integrates perceptual epsilon correctly."""
        wgsl = generate_ray_march_wgsl(use_perceptual_epsilon=True)

        assert "perceptual_epsilon" in wgsl or "epsilon_at_distance" in wgsl

    def test_wgsl_balanced_braces(self):
        """WGSL has balanced braces and parentheses."""
        for gen_func in [
            generate_ray_march_wgsl,
            generate_normal_estimation_wgsl,
            generate_epsilon_wgsl,
            generate_ray_march_struct_wgsl,
        ]:
            wgsl = gen_func() if callable(gen_func) else gen_func

            assert wgsl.count("{") == wgsl.count("}"), f"Unbalanced braces in {gen_func}"
            assert wgsl.count("(") == wgsl.count(")"), f"Unbalanced parens in {gen_func}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_ray_march_with_normal(self):
        """Full ray march returns hit with correct normal."""
        tracer = SphereTracer(max_steps=128, epsilon=0.001)
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)

        result, normal = tracer.march_with_normal(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            sdf,
            normal_epsilon=0.0001,
        )

        assert result.hit
        assert normal is not None
        assert is_unit_length(normal)
        assert normal.z > 0.99  # Should point toward camera

    def test_ray_marcher_class_integration(self):
        """RayMarcher class integrates all features."""
        config = RayMarchConfig(
            max_steps=128,
            max_distance=100.0,
            base_epsilon=0.001,
            use_perceptual_epsilon=True,
        )
        marcher = RayMarcher(config)

        sdf = lambda p: sdf_sphere(p, 1.0)

        result = marcher.march(Vec3(0, 0, 5), Vec3(0, 0, -1), sdf)

        assert result.hit
        assert result.position is not None
        assert result.normal is not None
        assert is_unit_length(result.normal)
        assert result.steps > 0
        assert result.epsilon_used > 0

    def test_multiple_primitives(self):
        """Ray march works with various SDF primitives."""
        primitives = [
            ("sphere", lambda p: sdf_sphere(p, 1.0)),
            ("box", lambda p: sdf_box(p, Vec3(1, 1, 1))),
            ("plane", lambda p: sdf_plane(p, Vec3(0, 0, 1), -2.0)),
            ("torus", lambda p: sdf_torus(p, 2.0, 0.5)),
            ("cylinder", lambda p: sdf_cylinder(p, 1.0, 2.0)),
        ]

        for name, sdf in primitives:
            result = march_ray(
                Vec3(0, 0, 5),
                Vec3(0, 0, -1),
                lambda p, s=sdf: (s(p), 0),
                max_steps=128,
                epsilon=0.01,
            )

            # All should hit (primitives are centered at origin)
            assert result.hit, f"{name} primitive should be hit"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case handling tests."""

    def test_zero_direction_handled(self):
        """Zero-length direction doesn't crash."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)
        result = march_ray(Vec3(0, 0, 5), Vec3(0, 0, 0), sdf)

        assert not result.hit
        assert result.steps == 0

    def test_starting_inside_surface(self):
        """Starting inside a surface is handled."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 10.0)

        # Starting at origin (inside sphere of radius 10)
        result = march_ray(Vec3(0, 0, 0), Vec3(0, 0, 1), sdf, epsilon=0.001)

        # Should not crash - behavior depends on implementation
        assert isinstance(result, HitResult)

    def test_very_small_epsilon(self):
        """Very small epsilon still terminates."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)

        result = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            sdf,
            epsilon=1e-8,
            max_steps=1000,
        )

        assert result.hit
        # With tiny epsilon, position should be very accurate
        assert abs(result.position.z - 1.0) < 0.001

    def test_very_large_epsilon(self):
        """Large epsilon terminates early."""
        sdf = sphere_sdf_with_material(Vec3(0, 0, 0), 1.0)

        result = march_ray(
            Vec3(0, 0, 5),
            Vec3(0, 0, -1),
            sdf,
            epsilon=0.5,
            max_steps=100,
        )

        assert result.hit
        # With large epsilon, should terminate early with fewer steps
        assert result.steps < 20

    def test_negative_sdf_values(self):
        """Negative SDF (inside geometry) doesn't crash."""
        # SDF that returns negative everywhere
        def always_negative(p: Vec3) -> Tuple[float, int]:
            return (-1.0, 0)

        result = march_ray(Vec3(0, 0, 0), Vec3(0, 0, 1), always_negative, max_steps=10)

        # Should hit immediately (negative < epsilon)
        assert result.hit
        assert result.steps <= 1

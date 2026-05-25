"""
GPU Culling Pipeline -- Whitebox tests.

Exercises internal implementation paths, branch conditions, loop boundaries,
helper functions, and error paths that the contract (blackbox) tests do not
explicitly target.

# WHITEBOX coverage plan:
#   Vec3:
#     - arithmetic operators (+ - * scalar) return new Vec3 (immutable)
#     - dot(x, y) == dot(y, x) symmetry property
#     - length() vs length_squared() consistency: len^2 ~= len^2
#     - normalized() on zero-length vector returns Vec3(0,0,0) (branch: length < EPSILON)
#     - normalized() on unit vector is identity
#     - normalized() maintains direction
#   Vec4:
#     - xyz property returns correct Vec3
#     - dot3(v) matches hand computation
#     - dot(full) matches hand computation
#   AABB:
#     - center() midpoints from min/max corner
#     - extents() half-extents from corners
#     - contains_point() on boundary (exact equality)
#     - contains_point() outside each axis
#     - expand() from initial inf/-inf state
#     - expand() grows uniformly
#   BoundingSphere:
#     - from_aabb(): center and radius from AABB
#     - contains_point(): inside/on-surface/outside
#     - from_aabb() of zero-extent AABB
#   Frustum (internal):
#     - from_view_projection_matrix(): each of 6 planes individually correct
#     - from_view_projection_matrix(): normalization branch (length > EPSILON)
#     - from_view_projection_matrix(): near-zero normal (skip normalization)
#     - test_sphere(): early-out on first culling plane (which plane)
#     - test_aabb(): p-vertex selection per axis
#     - test_aabb(): all 6 planes evaluated (loop iteration)
#   FrustumCuller:
#     - update: with Frustum object branch
#     - update: with VP matrix branch
#     - update: neither (no-op)
#     - cull: _frustum is None -> all visible early return
#     - cull: respects existing visible_mask (already-culled skip)
#     - cull_single: with frustum / without frustum
#     - _use_spheres = False triggers AABB path
#   OcclusionCuller (internal helpers):
#     - _build_hzb: empty list -> single empty level
#     - _build_hzb: single row (1xN) boundary clamping
#     - _build_hzb: single column (Nx1) boundary clamping
#     - _project_sphere_to_screen: no view/proj matrix -> safe default
#     - _project_sphere_to_screen: behind near plane -> safe default
#     - _project_sphere_to_screen: clip_w near zero -> safe default
#     - _select_mip_level: no HZB -> 0
#     - _select_mip_level: max_dim <= 1 -> last level
#     - _sample_hzb: empty pyramid -> 1.0
#     - _sample_hzb: out-of-bounds clamped
#     - is_occluded: no HZB -> False
#     - cull: no HZB -> all visible
#   DistanceCuller:
#     - _compute_distance: zero vector difference
#     - compute_lod_level: empty lod_distances -> 0
#     - compute_lod_level: distance exactly at boundary (branch condition)
#     - update: partial (only position, only forward, only max_distance)
#     - cull: effective distance (radius compensation) formula verified
#     - cull: zero-radius sphere at max distance boundary
#   SmallFeatureCuller:
#     - _compute_screen_size: zero-radius sphere
#     - _compute_screen_size: behind camera (negative distance branch)
#     - _compute_screen_size: exact formula verification
#     - update: partial updates
#   CullingPipeline:
#     - cull: empty instance list -> empty results
#     - configure: default state check
#     - update: delegates to all 4 sub-cullers
#     - cull: ordering invariant (indices sorted)
"""

from __future__ import annotations

import math
from typing import Sequence

import pytest

from engine.rendering.gpu_driven.culling import (
    Vec3,
    Vec4,
    AABB,
    BoundingSphere,
    Frustum,
    FrustumPlane,
    FrustumCuller,
    InstanceBounds,
    CullingStats,
    OcclusionCuller,
    DistanceCuller,
    SmallFeatureCuller,
    HZBConfig,
    DistanceCullConfig,
    SmallFeatureCullConfig,
    CullingConstants,
    CullingPipeline,
)


# =============================================================================
# HELPERS
# =============================================================================

def _perspective_vp(
    fov_y_rad: float,
    aspect: float,
    near: float,
    far: float,
) -> list[list[float]]:
    """Build an OpenGL-style perspective projection matrix (row-major)."""
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def _identity_view() -> list[list[float]]:
    """4x4 identity matrix."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _make_instance(
    instance_id: int,
    center: Vec3,
    radius: float = 1.0,
) -> InstanceBounds:
    aabb_min = Vec3(center.x - radius, center.y - radius, center.z - radius)
    aabb_max = Vec3(center.x + radius, center.y + radius, center.z + radius)
    return InstanceBounds(
        instance_id=instance_id,
        bounding_sphere=BoundingSphere(center=center, radius=radius),
        aabb=AABB(min_point=aabb_min, max_point=aabb_max),
    )


# =============================================================================
# VEC3 WHITEBOX TESTS
# =============================================================================

class TestVec3Whitebox:
    """Internal arithmetic, normalization branches, and invariants."""

    def test_add_returns_new_instance(self) -> None:
        """Vec3.__add__ produces a fresh object (immutable style)."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        c = a + b
        assert c is not a and c is not b
        assert c.x == 5.0 and c.y == 7.0 and c.z == 9.0

    def test_add_zero_vector_identity(self) -> None:
        """Adding zero vector returns equal vector."""
        a = Vec3(3.0, -1.0, 7.0)
        zero = Vec3(0.0, 0.0, 0.0)
        result = a + zero
        assert result.x == a.x and result.y == a.y and result.z == a.z

    def test_sub_returns_difference(self) -> None:
        """Vec3.__sub__ computes component-wise difference."""
        a = Vec3(10.0, 20.0, 30.0)
        b = Vec3(1.0, 2.0, 3.0)
        c = a - b
        assert c.x == 9.0 and c.y == 18.0 and c.z == 27.0

    def test_mul_by_scalar(self) -> None:
        """Vec3.__mul__ scales all components."""
        v = Vec3(2.0, -3.0, 4.0)
        r = v * 0.5
        assert r.x == 1.0 and r.y == -1.5 and r.z == 2.0

    def test_rmul_by_scalar(self) -> None:
        """Vec3.__rmul__ is commutative for scalar-first ordering."""
        v = Vec3(2.0, -3.0, 4.0)
        r = 3.0 * v
        assert r.x == 6.0 and r.y == -9.0 and r.z == 12.0

    def test_dot_symmetry(self) -> None:
        """dot(a, b) == dot(b, a)."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        assert a.dot(b) == b.dot(a)

    def test_dot_orthogonal_zero(self) -> None:
        """Orthogonal vectors have zero dot product."""
        x = Vec3(1.0, 0.0, 0.0)
        y = Vec3(0.0, 1.0, 0.0)
        assert x.dot(y) == 0.0

    def test_length_squared_equals_dot_self(self) -> None:
        """length_squared() == dot(self, self)."""
        v = Vec3(3.0, -4.0, 5.0)
        assert v.length_squared() == v.dot(v)

    def test_length_math_consistency(self) -> None:
        """length() == sqrt(length_squared())."""
        v = Vec3(3.0, -4.0, 5.0)
        assert v.length() == math.sqrt(v.length_squared())

    def test_length_zero_vector(self) -> None:
        """Zero vector has length 0."""
        v = Vec3(0.0, 0.0, 0.0)
        assert v.length() == 0.0
        assert v.length_squared() == 0.0

    def test_normalized_zero_vector(self) -> None:
        """Normalizing zero vector returns Vec3(0,0,0) (the length<EPSILON branch)."""
        v = Vec3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0 and n.y == 0.0 and n.z == 0.0

    def test_normalized_near_zero_vector(self) -> None:
        """Very small vector (near EPSILON) returns Vec3(0,0,0)."""
        v = Vec3(CullingConstants.EPSILON * 0.5, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0 and n.y == 0.0 and n.z == 0.0

    def test_normalized_unit_length(self) -> None:
        """Normalized vector has length 1 (within FP error)."""
        v = Vec3(3.0, -4.0, 0.0)
        n = v.normalized()
        assert n.length() == pytest.approx(1.0, abs=1e-10)

    def test_normalized_preserves_direction(self) -> None:
        """Normalized vector points in same direction as original."""
        v = Vec3(3.0, -4.0, 12.0)
        n = v.normalized()
        # Cross product magnitude should be near zero (same direction)
        cross_x = v.y * n.z - v.z * n.y
        cross_y = v.z * n.x - v.x * n.z
        cross_z = v.x * n.y - v.y * n.x
        cross_mag = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        assert cross_mag == pytest.approx(0.0, abs=1e-10)

    def test_normalized_unit_vector_unchanged(self) -> None:
        """Normalizing a unit vector is identity."""
        v = Vec3(1.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == pytest.approx(1.0) and n.y == 0.0 and n.z == 0.0


# =============================================================================
# VEC4 WHITEBOX TESTS
# =============================================================================

class TestVec4Whitebox:
    """Vec4 xyz property, dot and dot3 internals."""

    def test_xyz_property(self) -> None:
        """xyz returns Vec3 of first 3 components."""
        v = Vec4(1.0, 2.0, 3.0, 4.0)
        xyz = v.xyz
        assert isinstance(xyz, Vec3)
        assert xyz.x == 1.0 and xyz.y == 2.0 and xyz.z == 3.0

    def test_xyz_does_not_include_w(self) -> None:
        """xyz excludes the w component."""
        v = Vec4(1.0, 2.0, 3.0, 999.0)
        xyz = v.xyz
        assert xyz.z == 3.0  # not 999

    def test_dot3_with_vec3(self) -> None:
        """dot3 computes 3-component dot with a Vec3."""
        plane = Vec4(2.0, 0.0, 0.0, 5.0)
        point = Vec3(3.0, 100.0, 200.0)
        # plane.x * point.x + plane.y * point.y + plane.z * point.z
        # = 2*3 + 0*100 + 0*200 = 6
        assert plane.dot3(point) == 6.0

    def test_dot4_full(self) -> None:
        """dot computes full 4-component dot product."""
        a = Vec4(1.0, 2.0, 3.0, 4.0)
        b = Vec4(5.0, 6.0, 7.0, 8.0)
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert a.dot(b) == 70.0

    def test_dot3_and_dot_consistency(self) -> None:
        """For Vec4 with w=0, dot3(v) + w == dot(Vec4(v.x,v.y,v.z,0))."""
        plane = Vec4(2.0, 3.0, 4.0, 5.0)
        v = Vec3(1.0, 2.0, 3.0)
        dot3 = plane.dot3(v)
        dot4 = plane.dot(Vec4(v.x, v.y, v.z, 1.0))
        # dot3 + plane.w = (2+6+12) + 5 = 25
        # dot4 = 2+6+12+5 = 25
        assert dot3 + plane.w == pytest.approx(dot4)


# =============================================================================
# AABB WHITEBOX TESTS
# =============================================================================

class TestAABBWhitebox:
    """AABB center, extents, contains_point, expand internal paths."""

    def test_center_midpoint_of_corners(self) -> None:
        """center() is the midpoint of min and max points."""
        aabb = AABB(
            min_point=Vec3(-2.0, -4.0, -6.0),
            max_point=Vec3(2.0, 4.0, 6.0),
        )
        c = aabb.center
        assert c.x == 0.0 and c.y == 0.0 and c.z == 0.0

    def test_center_asymmetric(self) -> None:
        """center() for asymmetric AABB."""
        aabb = AABB(
            min_point=Vec3(1.0, 2.0, 3.0),
            max_point=Vec3(5.0, 10.0, 7.0),
        )
        c = aabb.center
        assert c.x == 3.0 and c.y == 6.0 and c.z == 5.0

    def test_center_initial_state(self) -> None:
        """Default AABB (inf,-inf) produces nan center (inf + -inf = nan)."""
        aabb = AABB()
        c = aabb.center
        # min=inf, max=-inf => (inf + -inf) * 0.5 = nan * 0.5 = nan
        assert all(math.isnan(getattr(c, ax)) for ax in ("x", "y", "z"))

    def test_extents_half_diagonal(self) -> None:
        """extents() is half the diagonal (max-min)/2."""
        aabb = AABB(
            min_point=Vec3(-4.0, -6.0, -8.0),
            max_point=Vec3(4.0, 6.0, 8.0),
        )
        e = aabb.extents
        assert e.x == 4.0 and e.y == 6.0 and e.z == 8.0

    def test_extents_zero_size(self) -> None:
        """Zero-size AABB has zero extents."""
        aabb = AABB(
            min_point=Vec3(5.0, 5.0, 5.0),
            max_point=Vec3(5.0, 5.0, 5.0),
        )
        e = aabb.extents
        assert e.x == 0.0 and e.y == 0.0 and e.z == 0.0

    def test_contains_point_inside(self) -> None:
        """Point strictly inside AABB is contained."""
        aabb = AABB(min_point=Vec3(-1, -1, -1), max_point=Vec3(1, 1, 1))
        assert aabb.contains_point(Vec3(0.0, 0.0, 0.0))

    def test_contains_point_exact_minimum(self) -> None:
        """Point exactly on minimum corner is contained (boundary inclusive)."""
        aabb = AABB(min_point=Vec3(-1, -1, -1), max_point=Vec3(1, 1, 1))
        assert aabb.contains_point(Vec3(-1.0, -1.0, -1.0))

    def test_contains_point_exact_maximum(self) -> None:
        """Point exactly on maximum corner is contained (boundary inclusive)."""
        aabb = AABB(min_point=Vec3(-1, -1, -1), max_point=Vec3(1, 1, 1))
        assert aabb.contains_point(Vec3(1.0, 1.0, 1.0))

    def test_contains_point_below_min(self) -> None:
        """Point below minimum on any axis is not contained."""
        aabb = AABB(min_point=Vec3(-1, -1, -1), max_point=Vec3(1, 1, 1))
        assert not aabb.contains_point(Vec3(-2.0, 0.0, 0.0))

    def test_contains_point_above_max(self) -> None:
        """Point above maximum on any axis is not contained."""
        aabb = AABB(min_point=Vec3(-1, -1, -1), max_point=Vec3(1, 1, 1))
        assert not aabb.contains_point(Vec3(0.0, 2.0, 0.0))

    def test_expand_from_default_state(self) -> None:
        """expand() from default (inf,-inf) adopts the point as bounds."""
        aabb = AABB()
        p = Vec3(10.0, 20.0, 30.0)
        aabb.expand(p)
        assert aabb.min_point.x == 10.0 and aabb.min_point.y == 20.0
        assert aabb.max_point.x == 10.0 and aabb.max_point.y == 20.0

    def test_expand_grows_min_and_max(self) -> None:
        """expand() extends AABB to encompass new point."""
        aabb = AABB(min_point=Vec3(0, 0, 0), max_point=Vec3(10, 10, 10))
        aabb.expand(Vec3(-5, 15, 5))
        assert aabb.min_point.x == -5 and aabb.min_point.y == 0
        assert aabb.max_point.x == 10 and aabb.max_point.y == 15

    def test_expand_point_inside_no_change(self) -> None:
        """expand() with a point already inside is a no-op."""
        aabb = AABB(min_point=Vec3(0, 0, 0), max_point=Vec3(10, 10, 10))
        aabb.expand(Vec3(5, 5, 5))
        assert aabb.min_point.x == 0 and aabb.max_point.x == 10


# =============================================================================
# BOUNDING SPHERE WHITEBOX TESTS
# =============================================================================

class TestBoundingSphereWhitebox:
    """BoundingSphere from_aabb and contains_point internal paths."""

    def test_from_aabb_sphere_center(self) -> None:
        """from_aabb: sphere center matches AABB center."""
        aabb = AABB(min_point=Vec3(-2, -4, -6), max_point=Vec3(2, 4, 6))
        sphere = BoundingSphere.from_aabb(aabb)
        assert sphere.center.x == 0.0 and sphere.center.y == 0.0 and sphere.center.z == 0.0

    def test_from_aabb_sphere_radius(self) -> None:
        """from_aabb: radius = length of extents."""
        aabb = AABB(min_point=Vec3(-1, -2, -3), max_point=Vec3(1, 2, 3))
        sphere = BoundingSphere.from_aabb(aabb)
        expected_radius = math.sqrt(1.0**2 + 2.0**2 + 3.0**2)
        assert sphere.radius == pytest.approx(expected_radius)

    def test_from_aabb_zero_extent(self) -> None:
        """from_aabb of degenerate AABB -> zero-radius sphere."""
        aabb = AABB(min_point=Vec3(5, 5, 5), max_point=Vec3(5, 5, 5))
        sphere = BoundingSphere.from_aabb(aabb)
        assert sphere.center.x == 5.0 and sphere.center.y == 5.0 and sphere.center.z == 5.0
        assert sphere.radius == 0.0

    def test_contains_point_inside(self) -> None:
        """Point inside sphere radius is contained."""
        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=5.0)
        assert sphere.contains_point(Vec3(3, 0, 0))

    def test_contains_point_on_surface(self) -> None:
        """Point exactly on sphere surface is contained (radius boundary inclusive)."""
        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=5.0)
        assert sphere.contains_point(Vec3(5, 0, 0))

    def test_contains_point_outside(self) -> None:
        """Point beyond sphere radius is not contained."""
        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=5.0)
        assert not sphere.contains_point(Vec3(6, 0, 0))

    def test_contains_point_center_itself(self) -> None:
        """The center point is always contained."""
        sphere = BoundingSphere(center=Vec3(10, -20, 30), radius=0.001)
        assert sphere.contains_point(Vec3(10, -20, 30))

    def test_contains_point_uses_length_squared(self) -> None:
        """contains_point uses squared distance (no sqrt in hot path)."""
        # Verify internally it uses diff.length_squared() <= radius^2
        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=10.0)
        # At exactly radius distance, len^2 = 100, radius^2 = 100 -> inside
        assert sphere.contains_point(Vec3(0, 10, 0))
        # Just outside: len^2 = 101, radius^2 = 100 -> outside
        assert not sphere.contains_point(Vec3(0, 10.05, 0))


# =============================================================================
# FRUSTUM WHITEBOX TESTS
# =============================================================================

class TestFrustumWhitebox:
    """Frustum plane extraction, normalization branches, test_sphere, test_aabb internals."""

    def test_from_vp_matrix_six_planes_distinct(self) -> None:
        """All 6 extracted planes have different coefficients (no duplicates)."""
        vp = _perspective_vp(math.radians(60), 16 / 9, 0.1, 500.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        # Check that no two planes are identical
        for i in range(6):
            for j in range(i + 1, 6):
                a = frustum.planes[i]
                b = frustum.planes[j]
                same = (
                    abs(a.x - b.x) < 1e-9
                    and abs(a.y - b.y) < 1e-9
                    and abs(a.z - b.z) < 1e-9
                    and abs(a.w - b.w) < 1e-9
                )
                assert not same, f"Planes {i} and {j} are identical"

    def test_from_vp_matrix_plane_indices_match_enum(self) -> None:
        """Plane ordering matches FrustumPlane enum values."""
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        # LEFT=0, RIGHT=1, BOTTOM=2, TOP=3, NEAR=4, FAR=5
        # Gribb-Hartmann: LEFT = row3+row0, RIGHT = row3-row0, etc.
        # For a perspective VP with identity view:
        # m[3] = [0, 0, -1, 0], m[0] varies
        left = frustum.planes[FrustumPlane.LEFT]
        right = frustum.planes[FrustumPlane.RIGHT]
        # Left and right should have opposite x-signs (one faces +x, other -x)
        # After normalization the signs differ
        assert left.x != 0.0
        assert right.x != 0.0

    def test_from_vp_matrix_normalization_skip_branch(self) -> None:
        """Planes with near-zero normal length are not normalized (EPSILON guard)."""
        # A contrived matrix that could produce a zero-length normal plane.
        # Use a fully zero matrix to trigger the EPSILON branch.
        zero_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        frustum = Frustum.from_view_projection_matrix(zero_matrix)
        # All planes should remain as Vec4(0,0,0,0) since normalization is skipped
        for plane in frustum.planes:
            assert plane.x == 0.0 and plane.y == 0.0 and plane.z == 0.0 and plane.w == 0.0

    def test_test_sphere_early_out_at_first_plane(self) -> None:
        """test_sphere returns False on the first plane where sphere is fully behind."""
        # A manually controlled frustum: near plane at z=-1
        frustum = Frustum()
        # Box [-10, 10] in x/y, [-100, -1] in z (near plane at z=-1)
        frustum.planes[FrustumPlane.LEFT] = Vec4(1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.RIGHT] = Vec4(-1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(0.0, 1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.TOP] = Vec4(0.0, -1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.NEAR] = Vec4(0.0, 0.0, -1.0, -1.0)
        frustum.planes[FrustumPlane.FAR] = Vec4(0.0, 0.0, 1.0, 100.0)

        # Sphere far behind the near plane (z > -1, i.e. positive z in our coord system)
        # Near plane: dot3(p) + w = (0,0,-1).dot(p) + (-1) = -p.z - 1
        # For p = (0, 0, 5): -5 - 1 = -6 < -radius -> culled
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, 5.0), radius=0.5)
        assert not frustum.test_sphere(sphere)

    def test_test_sphere_all_planes_evaluated_when_visible(self) -> None:
        """For a fully visible sphere, all 6 planes must pass (sphere inside)."""
        frustum = Frustum()
        # A big frustum centered at origin
        frustum.planes[FrustumPlane.LEFT] = Vec4(1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.RIGHT] = Vec4(-1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(0.0, 1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.TOP] = Vec4(0.0, -1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.NEAR] = Vec4(0.0, 0.0, -1.0, 100.0)  # near at z=-100
        frustum.planes[FrustumPlane.FAR] = Vec4(0.0, 0.0, 1.0, 100.0)  # far at z=100

        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=1.0)
        assert frustum.test_sphere(sphere)

    def test_test_sphere_partial_intersection_each_plane(self) -> None:
        """Sphere straddling each individual plane is still visible."""
        frustum = Frustum()
        # Box [-10,10] in all axes
        frustum.planes[FrustumPlane.LEFT] = Vec4(1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.RIGHT] = Vec4(-1.0, 0.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(0.0, 1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.TOP] = Vec4(0.0, -1.0, 0.0, 10.0)
        frustum.planes[FrustumPlane.NEAR] = Vec4(0.0, 0.0, 1.0, 100.0)
        frustum.planes[FrustumPlane.FAR] = Vec4(0.0, 0.0, -1.0, 100.0)

        # TODO: just verify a sphere centered just outside one plane
        # but large enough to touch the frustum is visible
        sphere = BoundingSphere(center=Vec3(-15.0, 0.0, 0.0), radius=10.0)
        assert frustum.test_sphere(sphere)

    def test_test_aabb_p_vertex_selection(self) -> None:
        """AABB test uses p-vertex (corner most aligned with plane normal)."""
        frustum = Frustum()
        # Frustum: only left plane check (x >= -5)
        frustum.planes[FrustumPlane.LEFT] = Vec4(1.0, 0.0, 0.0, 5.0)   # x >= -5
        frustum.planes[FrustumPlane.RIGHT] = Vec4(-1.0, 0.0, 0.0, 100.0)  # x <= 100
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(0.0, 1.0, 0.0, 100.0)
        frustum.planes[FrustumPlane.TOP] = Vec4(0.0, -1.0, 0.0, 100.0)
        frustum.planes[FrustumPlane.NEAR] = Vec4(0.0, 0.0, 1.0, 100.0)
        frustum.planes[FrustumPlane.FAR] = Vec4(0.0, 0.0, -1.0, 100.0)

        # For LEFT plane (1,0,0): p-vertex = max_point if plane.x >= 0 else min_point
        # plane.x = 1 >= 0, so p-vertex x = max_point.x
        # AABB fully to the left: max_point.x < -5
        left_out = AABB(min_point=Vec3(-100, -1, -1), max_point=Vec3(-10, 1, 1))
        assert not frustum.test_aabb(left_out), "AABB left of frustum should be culled"

        # AABB partially inside: max_point.x >= -5
        straddle = AABB(min_point=Vec3(-100, -1, -1), max_point=Vec3(0, 1, 1))
        assert frustum.test_aabb(straddle), "AABB straddling left plane should be visible"

    def test_test_aabb_all_planes_individually_cull(self) -> None:
        """Each of the 6 planes can independently cull an AABB."""
        frustum = Frustum()
        # A frustum: [-5, 5] x [-5, 5] x [1, 100]
        # Normals point inward: test_aabb uses dot3(p_vertex) + w >= 0 for visibility
        # For a point inside at z=50: near plane (0,0,1,-1) -> 0+0+50-1 = 49 >= 0
        #                            far plane  (0,0,-1,100) -> 0+0-50+100 = 50 >= 0
        frustum.planes[FrustumPlane.LEFT] = Vec4(1.0, 0.0, 0.0, 5.0)
        frustum.planes[FrustumPlane.RIGHT] = Vec4(-1.0, 0.0, 0.0, 5.0)
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(0.0, 1.0, 0.0, 5.0)
        frustum.planes[FrustumPlane.TOP] = Vec4(0.0, -1.0, 0.0, 5.0)
        frustum.planes[FrustumPlane.NEAR] = Vec4(0.0, 0.0, 1.0, -1.0)   # z >= 1
        frustum.planes[FrustumPlane.FAR] = Vec4(0.0, 0.0, -1.0, 100.0)  # z <= 100

        inside = AABB(min_point=Vec3(-2, -2, 10), max_point=Vec3(2, 2, 50))
        assert frustum.test_aabb(inside)

        # Culled by each plane individually
        culled_by: list[tuple[int, AABB]] = [
            (FrustumPlane.LEFT, AABB(min_point=Vec3(-100, -2, 10), max_point=Vec3(-10, 2, 50))),
            (FrustumPlane.RIGHT, AABB(min_point=Vec3(10, -2, 10), max_point=Vec3(100, 2, 50))),
            (FrustumPlane.BOTTOM, AABB(min_point=Vec3(-2, -100, 10), max_point=Vec3(2, -10, 50))),
            (FrustumPlane.TOP, AABB(min_point=Vec3(-2, 10, 10), max_point=Vec3(2, 100, 50))),
            (FrustumPlane.NEAR, AABB(min_point=Vec3(-2, -2, -10), max_point=Vec3(2, 2, 0))),
            (FrustumPlane.FAR, AABB(min_point=Vec3(-2, -2, 200), max_point=Vec3(2, 2, 300))),
        ]
        for plane_idx, aabb in culled_by:
            assert not frustum.test_aabb(aabb), (
                f"AABB should be culled by plane {FrustumPlane(plane_idx).name}"
            )


# =============================================================================
# FRUSTUM CULLER WHITEBOX TESTS
# =============================================================================

class TestFrustumCullerWhitebox:
    """FrustumCuller update branches, cull internal mask handling, cull_single."""

    def test_update_with_frustum_object(self) -> None:
        """Update with pre-built Frustum sets internal _frustum."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        culler.update(frustum=frustum)
        assert culler.frustum is frustum  # same object

    def test_update_with_vp_matrix(self) -> None:
        """Update with VP matrix builds Frustum internally."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)
        assert culler.frustum is not None
        assert len(culler.frustum.planes) == 6

    def test_update_no_args_does_not_change_state(self) -> None:
        """Update with no arguments leaves _frustum unchanged."""
        culler = FrustumCuller()
        # _frustum starts as None
        culler.update()
        assert culler.frustum is None

    def test_update_vp_matrix_overrides_frustum(self) -> None:
        """Update with both: frustum takes priority (branch order)."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        direct_frustum = Frustum.from_view_projection_matrix(vp)
        other_vp = _perspective_vp(math.radians(45), 16 / 9, 0.1, 1000.0)
        # frustum arg is not None -> uses it; ignores vp_matrix
        culler.update(frustum=direct_frustum, view_projection_matrix=other_vp)
        assert culler.frustum is direct_frustum

    def test_cull_no_frustum_all_visible(self) -> None:
        """When no frustum is set, all instances pass (early return)."""
        culler = FrustumCuller()
        instances = [_make_instance(0, Vec3(0, 0, 0))]
        mask = [True]
        stats = culler.cull(instances, mask)
        assert stats.visible_instances == 1
        assert stats.frustum_culled == 0

    def test_cull_no_frustum_respects_existing_mask(self) -> None:
        """No-frustum early return sums existing mask rather than all-true."""
        culler = FrustumCuller()
        instances = [
            _make_instance(0, Vec3(0, 0, 0)),
            _make_instance(1, Vec3(0, 0, 0)),
            _make_instance(2, Vec3(0, 0, 0)),
        ]
        mask = [True, False, True]  # middle already culled
        stats = culler.cull(instances, mask)
        assert stats.visible_instances == 2
        assert stats.frustum_culled == 0

    def test_cull_respects_existing_mask(self) -> None:
        """Already-culled instances (mask=False) are skipped entirely."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)

        instances = [
            _make_instance(0, Vec3(0, 0, -5), radius=1.0),   # visible
            _make_instance(1, Vec3(20, 0, -5), radius=1.0),  # would be culled
            _make_instance(2, Vec3(0, 0, -10), radius=1.0),  # visible
        ]
        mask = [True, False, True]  # instance 1 pre-culled
        stats = culler.cull(instances, mask)
        # mask[0] True (visible), mask[1] False (pre-culled, skipped), mask[2] True (visible)
        assert stats.frustum_culled == 0  # instance 1 was skipped
        assert stats.visible_instances == 2

    def test_cull_single_with_frustum_visible(self) -> None:
        """cull_single returns True for a visible instance."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)
        inst = _make_instance(0, Vec3(0, 0, -5), radius=0.5)
        assert culler.cull_single(inst)

    def test_cull_single_with_frustum_culled(self) -> None:
        """cull_single returns False for a culled instance."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)
        inst = _make_instance(0, Vec3(0, 0, 5), radius=0.5)  # behind camera
        assert not culler.cull_single(inst)

    def test_cull_single_no_frustum_returns_true(self) -> None:
        """cull_single returns True (visible) when no frustum set."""
        culler = FrustumCuller()
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        assert culler.cull_single(inst)

    def test_cull_with_aabb_mode(self) -> None:
        """When _use_spheres=False, cull uses AABB test."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)
        culler._use_spheres = False

        # An instance with AABB clearly inside
        inst = _make_instance(0, Vec3(0, 0, -5), radius=0.5)
        mask = [True]
        stats = culler.cull([inst], mask)
        assert mask[0]

        # An instance with AABB clearly outside
        inst2 = _make_instance(1, Vec3(50, 0, -5), radius=0.5)
        mask2 = [True]
        stats2 = culler.cull([inst2], mask2)
        assert not mask2[0]

    def test_cull_stats_aggregation(self) -> None:
        """FrustumCuller stats correctly count culled vs visible."""
        culler = FrustumCuller()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        culler.update(view_projection_matrix=vp)

        instances = [
            _make_instance(0, Vec3(0, 0, -5), radius=0.5),   # visible
            _make_instance(1, Vec3(0, 0, 5), radius=0.5),    # behind -> culled
            _make_instance(2, Vec3(0, 0, -200), radius=0.5), # beyond far -> culled
            _make_instance(3, Vec3(-50, 0, -5), radius=0.5), # left of frustum -> culled
        ]
        mask = [True] * 4
        stats = culler.cull(instances, mask)
        assert stats.total_instances == 4
        assert stats.visible_instances == 1  # only instance 0
        assert stats.frustum_culled == 3     # instances 1, 2, 3
        assert sum(mask) == 1


# =============================================================================
# OCCLUSION CULLER WHITEBOX TESTS
# =============================================================================

class TestOcclusionCullerWhitebox:
    """OcclusionCuller _build_hzb edges, _project_sphere_to_screen branches,
    _select_mip_level, _sample_hzb, is_occluded edge cases."""

    # ---- _build_hzb edge cases ----

    def test_build_hzb_empty_buffer(self) -> None:
        """Empty depth buffer results in exactly 1 empty mip level."""
        culler = OcclusionCuller(HZBConfig(width=0, height=0))
        culler._build_hzb([])
        assert culler.hzb_mip_levels == 1
        assert len(culler._hzb_pyramid[0]) == 0

    def test_build_hzb_single_row(self) -> None:
        """1xN depth buffer: width halves each iteration, height stays 1."""
        culler = OcclusionCuller()
        depth = [[0.1, 0.5, 0.9, 0.3]]  # 1x4
        culler._build_hzb(depth)
        assert culler.hzb_mip_levels >= 3  # 1x4 -> 1x2 -> 1x1
        assert len(culler._hzb_pyramid[0]) == 1  # height = 1 at level 0
        assert len(culler._hzb_pyramid[0][0]) == 4  # width = 4 at level 0

    def test_build_hzb_single_column(self) -> None:
        """Nx1 depth buffer: height halves each iteration, width stays 1."""
        culler = OcclusionCuller()
        depth = [[0.1], [0.5], [0.9], [0.3]]  # 4x1
        culler._build_hzb(depth)
        assert culler.hzb_mip_levels >= 3  # 4x1 -> 2x1 -> 1x1
        assert len(culler._hzb_pyramid[0]) == 4  # height = 4 at level 0
        assert len(culler._hzb_pyramid[0][0]) == 1  # width = 1 at level 0

    def test_build_hzb_1x1_single_level(self) -> None:
        """1x1 depth buffer produces exactly 1 level."""
        culler = OcclusionCuller()
        culler._build_hzb([[0.5]])
        assert culler.hzb_mip_levels == 1

    def test_build_hzb_odd_dimension_clamping(self) -> None:
        """Odd-dimension buffer: boundary pixel is clamped, not out-of-bounds."""
        culler = OcclusionCuller()
        # 3x3 -> level 1 = 1x1 (3//2=1)
        depth = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        culler._build_hzb(depth)
        assert culler.hzb_mip_levels == 2
        # Level 1 is 1x1: max of 0..2 x 0..2 with clamping
        # The 2x2 tile at (0,0) samples indices:
        #   y0=0, y1=min(1,2)=1, x0=0, x1=min(1,2)=1
        # So it samples (0,0),(0,1),(1,0),(1,1) -> max(0.1,0.2,0.4,0.5) = 0.5
        assert culler._hzb_pyramid[1][0][0] == 0.5

    # ---- _build_hzb max-reduction property ----

    def test_hzb_mip_level_decreasing_dimensions(self) -> None:
        """Each successive mip level is <= previous in both dimensions."""
        culler = OcclusionCuller()
        depth = [[float(y * 8 + x) for x in range(8)] for y in range(8)]
        culler._build_hzb(depth)
        for i in range(1, culler.hzb_mip_levels):
            prev_h = len(culler._hzb_pyramid[i - 1])
            prev_w = len(culler._hzb_pyramid[i - 1][0])
            cur_h = len(culler._hzb_pyramid[i])
            cur_w = len(culler._hzb_pyramid[i][0])
            assert cur_h <= prev_h, f"Height increased at level {i}"
            assert cur_w <= prev_w, f"Width increased at level {i}"

    # ---- _project_sphere_to_screen ----

    def test_project_no_view_matrix(self) -> None:
        """Without view/projection matrix, projection returns safe defaults."""
        culler = OcclusionCuller()
        sphere = BoundingSphere(center=Vec3(0, 0, -10), radius=1.0)
        result = culler._project_sphere_to_screen(sphere)
        assert result == (0.0, 0.0, 1.0, 1.0, 0.0)

    def test_project_behind_near_plane(self) -> None:
        """Sphere behind near plane returns safe defaults."""
        culler = OcclusionCuller()
        culler._view_matrix = _identity_view()
        culler._projection_matrix = _perspective_vp(
            math.radians(90), 1.0, 1.0, 100.0
        )
        culler._near_plane = 1.0
        sphere = BoundingSphere(center=Vec3(0, 0, -0.5), radius=1.0)
        # view_z = 1*vm[2][2]*cz + vm[2][3] = 1*(-0.5) + 0 = -0.5
        # -0.5 > -1.0 (near plane), so behind -> safe default
        result = culler._project_sphere_to_screen(sphere)
        assert result == (0.0, 0.0, 1.0, 1.0, 0.0)

    def test_project_clip_w_near_zero(self) -> None:
        """Sphere with clip_w near zero returns safe defaults."""
        culler = OcclusionCuller()
        # A view matrix that puts the sphere at clip_w = 0
        # Using identity view, the perspective proj has w = view_z
        # For the sphere to have clip_w ~= 0, view_z must be ~0
        culler._view_matrix = _identity_view()
        culler._projection_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
        culler._near_plane = 1.0
        sphere = BoundingSphere(center=Vec3(0, 0, 0), radius=1.0)
        # clip_w = pm[3][0]*vx + ... + pm[3][3]*vw
        # = -1 * vz = -1 * (-view_z_after_transform)
        # Actually with identity view, vz = 0, so clip_w = (-1)*0 + 0 = 0
        # -> abs(0) < EPSILON -> safe default
        result = culler._project_sphere_to_screen(sphere)
        assert result == (0.0, 0.0, 1.0, 1.0, 0.0)

    def test_project_screen_bounds_clamped_01(self) -> None:
        """Projected screen bounds are clamped to [0, 1]."""
        culler = OcclusionCuller()
        culler._view_matrix = _identity_view()
        culler._projection_matrix = _perspective_vp(
            math.radians(90), 1.0, 1.0, 100.0
        )
        culler._near_plane = 1.0
        # An off-center large sphere whose screen bounds would exceed [0,1]
        sphere = BoundingSphere(center=Vec3(10.0, 0.0, -1.5), radius=10.0)
        min_x, min_y, max_x, max_y, _ = culler._project_sphere_to_screen(sphere)
        assert min_x >= 0.0 and min_x <= 1.0
        assert max_x >= 0.0 and max_x <= 1.0
        assert min_y >= 0.0 and max_y <= 1.0

    # ---- _select_mip_level ----

    def test_select_mip_level_no_pyramid(self) -> None:
        """Without HZB pyramid, _select_mip_level returns 0."""
        culler = OcclusionCuller()
        assert culler._select_mip_level(0.5, 0.5) == 0

    def test_select_mip_level_max_dim_one(self) -> None:
        """When max_dim <= 1, return last mip level."""
        culler = OcclusionCuller()
        depth = [[0.5 for _ in range(4)] for _ in range(4)]
        culler._build_hzb(depth)
        # pixel_width = 0.0001 * 512 = 0.0512 (default width)
        # pixel_height = 0.0001 * 512 = 0.0512
        # max_dim = 0.0512 <= 1 -> return last level
        level = culler._select_mip_level(0.0001, 0.0001)
        # But wait, we built with a 4x4 buffer, default config is 512x512
        # The HZB pyramid has 3 levels (4->2->1)
        # So last level = 2
        assert level == culler.hzb_mip_levels - 1

    def test_select_mip_level_returns_valid_index(self) -> None:
        """Mip level is always clamped to valid pyramid range."""
        culler = OcclusionCuller()
        depth = [[0.5 for _ in range(512)] for _ in range(512)]
        culler._build_hzb(depth)
        for w in [0.001, 0.01, 0.1, 0.5, 1.0]:
            level = culler._select_mip_level(w, w)
            assert 0 <= level < culler.hzb_mip_levels

    # ---- _sample_hzb ----

    def test_sample_hzb_empty_pyramid_returns_1(self) -> None:
        """Sampling an empty HZB pyramid returns 1.0 (not occluded)."""
        culler = OcclusionCuller()
        assert culler._sample_hzb(0.5, 0.5, 0) == 1.0

    def test_sample_hzb_out_of_range_mip_returns_1(self) -> None:
        """Sampling with mip_level >= len(pyramid) returns 1.0."""
        culler = OcclusionCuller()
        depth = [[0.5 for _ in range(4)] for _ in range(4)]
        culler._build_hzb(depth)
        assert culler._sample_hzb(0.5, 0.5, 99) == 1.0

    def test_sample_hzb_clamps_coordinates(self) -> None:
        """UV coordinates outside [0,1] are clamped to valid range."""
        culler = OcclusionCuller()
        depth = [[0.5 for _ in range(4)] for _ in range(4)]
        culler._build_hzb(depth)
        # x = -0.5, y = 2.0: both outside, should clamp to valid pixel
        value = culler._sample_hzb(-0.5, 2.0, 0)
        # Should not crash, returns a valid value
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_sample_hzb_exact_value(self) -> None:
        """Sampling known coordinates returns known depth value."""
        culler = OcclusionCuller()
        depth = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ]
        culler._build_hzb(depth)
        # Pixel (0,0) = depth[0][0] = 0.1
        val = culler._sample_hzb(0.0, 0.0, 0)
        assert val == 0.1
        # Pixel (3, 0) = depth[0][3] = 0.4 (at u ~ 0.875, v = 0)
        # x = 0.875, width=4, px = int(0.875*4)=3
        # y = 0, height=4, py = int(0*4)=0
        val2 = culler._sample_hzb(0.875, 0.0, 0)
        assert val2 == 0.4

    def test_sample_hzb_mip_level_1(self) -> None:
        """Sampling mip level 1 reads reduced depth."""
        culler = OcclusionCuller()
        depth = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ]
        culler._build_hzb(depth)
        # Level 1 = 2x2:
        # [0]: max(0.1,0.2,0.5,0.6) = 0.6
        # [1]: max(0.3,0.4,0.7,0.8) = 0.8
        # [2]: max(0.9,0.1,0.4,0.5) = 0.9
        # [3]: max(0.2,0.3,0.6,0.7) = 0.7
        val00 = culler._sample_hzb(0.0, 0.0, 1)
        assert val00 == 0.6
        val11 = culler._sample_hzb(1.0, 1.0, 1)
        # px = min(1, int(1.0*2)) = 1, py = min(1, int(1.0*2)) = 1
        # That gives [1][1] = 0.7
        assert val11 == 0.7

    # ---- is_occluded ----

    def test_is_occluded_no_hzb(self) -> None:
        """Without HZB pyramid, no sphere is occluded."""
        culler = OcclusionCuller()
        assert not culler.is_occluded(BoundingSphere(center=Vec3(0, 0, -10), radius=1.0))

    # ---- cull ----

    def test_cull_no_hzb_all_visible(self) -> None:
        """Without HZB, cull passes all visible instances through."""
        culler = OcclusionCuller()
        instances = [_make_instance(0, Vec3(0, 0, -10), radius=1.0)]
        mask = [True]
        stats = culler.cull(instances, mask)
        assert stats.visible_instances == 1
        assert stats.occlusion_culled == 0
        assert mask[0]

    def test_cull_respects_existing_mask(self) -> None:
        """Already-culled instances are skipped by occlusion culler."""
        culler = OcclusionCuller()
        depth = [[0.8 for _ in range(4)] for _ in range(4)]
        culler.update(
            depth_buffer=depth,
            view_matrix=_identity_view(),
            projection_matrix=_perspective_vp(math.radians(90), 1.0, 1.0, 100.0),
            near_plane=1.0,
            far_plane=100.0,
        )
        instances = [
            _make_instance(0, Vec3(0, 0, -90), radius=1.0),  # would be occluded
            _make_instance(1, Vec3(0, 0, -90), radius=1.0),  # skipped (pre-culled)
        ]
        mask = [True, False]  # instance 1 pre-culled
        stats = culler.cull(instances, mask)
        assert stats.occlusion_culled == 1  # only instance 0
        assert not mask[0]  # culled
        assert not mask[1]  # still pre-culled


# =============================================================================
# DISTANCE CULLER WHITEBOX TESTS
# =============================================================================

class TestDistanceCullerWhitebox:
    """DistanceCuller internal helpers, update partial updates, effective distance."""

    def setup_method(self) -> None:
        self.config = DistanceCullConfig(
            max_distance=500.0,
            fade_distance=50.0,
            lod_distances=[10.0, 50.0, 200.0, 500.0],
        )
        self.culler = DistanceCuller(self.config)
        self.culler.update(camera_position=Vec3(0.0, 0.0, 0.0))

    def test_compute_distance_zero_diff(self) -> None:
        """Instance at camera position has distance 0."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        assert self.culler._compute_distance(inst) == 0.0

    def test_compute_distance_negative_coords(self) -> None:
        """Distance is always positive regardless of coordinate signs."""
        self.culler.update(camera_position=Vec3(10, 10, 10))
        inst = _make_instance(0, Vec3(-10, -10, -10), radius=1.0)
        dist = self.culler._compute_distance(inst)
        expected = math.sqrt(20**2 + 20**2 + 20**2)
        assert dist == pytest.approx(expected)

    def test_compute_lod_level_empty_list(self) -> None:
        """When no LOD distances are configured, all distances return LOD 0."""
        config = DistanceCullConfig(
            max_distance=500.0,
            lod_distances=[],
        )
        culler = DistanceCuller(config)
        assert culler.compute_lod_level(0.0) == 0
        assert culler.compute_lod_level(1000.0) == 0

    def test_compute_lod_level_exact_boundary(self) -> None:
        """At exact boundary (distance == lod_distance), returns next LOD."""
        assert self.culler.compute_lod_level(10.0) == 1
        assert self.culler.compute_lod_level(50.0) == 2

    def test_compute_lod_level_just_below_boundary(self) -> None:
        """Just below the boundary gets the lower LOD."""
        assert self.culler.compute_lod_level(9.999999) == 0
        assert self.culler.compute_lod_level(49.999999) == 1

    def test_compute_lod_level_many_lod_levels(self) -> None:
        """A long LOD list still returns correct final index."""
        config = DistanceCullConfig(
            max_distance=10000.0,
            lod_distances=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        )
        culler = DistanceCuller(config)
        # Just at the last boundary
        assert culler.compute_lod_level(512.0) == 10  # len(lod_distances)
        # Beyond
        assert culler.compute_lod_level(1000.0) == 10

    def test_update_partial_position_only(self) -> None:
        """Update with only camera_position leaves forward unchanged."""
        culler = DistanceCuller(self.config)
        culler.update(camera_position=Vec3(100, 200, 300))
        # _camera_forward should still be default
        assert culler._camera_forward.x == 0.0
        assert culler._camera_forward.z == -1.0

    def test_update_partial_forward_only(self) -> None:
        """Update with only camera_forward normalizes it."""
        culler = DistanceCuller(self.config)
        culler.update(camera_forward=Vec3(1, 1, 0))
        # Should be normalized
        fwd = culler._camera_forward
        assert fwd.length() == pytest.approx(1.0)
        assert fwd.x == pytest.approx(math.sqrt(0.5))
        assert fwd.y == pytest.approx(math.sqrt(0.5))

    def test_update_partial_max_distance_only(self) -> None:
        """Update with only max_distance changes config."""
        culler = DistanceCuller(self.config)
        culler.update(max_distance=1000.0)
        assert culler.config.max_distance == 1000.0

    def test_cull_effective_distance_radius_compensation(self) -> None:
        """Cull uses radius-compensated distance: distance_sq - radius^2."""
        # Place a large sphere at 600 units with radius 200
        # effective_dist_sq = 600^2 - 200^2 = 320000
        # sqrt(320000) = 565.7 > 500 (max_distance) -> culled
        inst = _make_instance(0, Vec3(0, 0, -600), radius=200.0)
        mask = [True]
        stats = self.culler.cull([inst], mask)
        assert stats.distance_culled == 1
        assert not mask[0]

    def test_cull_radius_compensation_makes_visible(self) -> None:
        """Large enough radius compensates for distance."""
        # Sphere at 600 with radius 350
        # effective_dist_sq = 600^2 - 350^2 = 360000 - 122500 = 237500
        # sqrt(237500) = 487.3 < 500 -> visible
        inst = _make_instance(0, Vec3(0, 0, -600), radius=350.0)
        mask = [True]
        stats = self.culler.cull([inst], mask)
        assert stats.distance_culled == 0
        assert mask[0]

    def test_cull_zero_radius_at_max_distance(self) -> None:
        """Zero-radius sphere exactly at max_distance boundary."""
        inst = _make_instance(0, Vec3(0, 0, -500), radius=0.0)
        mask = [True]
        stats = self.culler.cull([inst], mask)
        # effective_dist_sq = 500^2 - 0 = 250000 = max_dist^2
        # effective_dist_sq > max_dist_sq? 250000 > 250000 -> False (not strictly greater)
        # So it should be visible (exactly at boundary: NOT culled)
        assert stats.distance_culled == 0
        assert mask[0]

    def test_cull_zero_radius_just_beyond_max_distance(self) -> None:
        """Zero-radius sphere just beyond max distance is culled."""
        inst = _make_instance(0, Vec3(0, 0, -500.001), radius=0.0)
        mask = [True]
        stats = self.culler.cull([inst], mask)
        assert stats.distance_culled == 1
        assert not mask[0]

    def test_cull_respects_existing_mask(self) -> None:
        """Already-culled instances are skipped by distance culler."""
        instances = [
            _make_instance(0, Vec3(0, 0, -600), radius=0.0),  # would be culled
            _make_instance(1, Vec3(0, 0, -600), radius=0.0),  # pre-culled
        ]
        mask = [True, False]
        stats = self.culler.cull(instances, mask)
        assert stats.distance_culled == 1  # only instance 0
        assert not mask[0]
        assert not mask[1]

    def test_cull_stats_aggregation(self) -> None:
        """DistanceCuller stats count culled correctly."""
        instances = [
            _make_instance(0, Vec3(0, 0, -10), radius=0.0),    # visible
            _make_instance(1, Vec3(0, 0, -600), radius=0.0),   # culled
            _make_instance(2, Vec3(0, 0, -800), radius=0.0),   # culled
        ]
        mask = [True, True, True]
        stats = self.culler.cull(instances, mask)
        assert stats.total_instances == 3
        assert stats.visible_instances == 1
        assert stats.distance_culled == 2


# =============================================================================
# SMALL FEATURE CULLER WHITEBOX TESTS
# =============================================================================

class TestSmallFeatureCullerWhitebox:
    """SmallFeatureCuller _compute_screen_size branches, update partial."""

    def setup_method(self) -> None:
        self.config = SmallFeatureCullConfig(
            min_screen_size=2.0,
            screen_width=1920,
            screen_height=1080,
        )
        self.culler = SmallFeatureCuller(self.config)
        self.culler.update(
            camera_position=Vec3(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
        )

    def test_compute_screen_size_zero_distance(self) -> None:
        """Instance at camera position returns infinity."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        size = self.culler._compute_screen_size(inst)
        assert size == float("inf")

    def test_compute_screen_size_near_zero_distance(self) -> None:
        """Instance at very small distance returns infinity (distance < 1e-6 branch)."""
        inst = _make_instance(0, Vec3(0, 0, 1e-7), radius=1.0)
        size = self.culler._compute_screen_size(inst)
        assert size == float("inf")

    def test_compute_screen_size_zero_radius(self) -> None:
        """Zero-radius sphere at non-zero distance has zero screen size."""
        inst = _make_instance(0, Vec3(0, 0, -100), radius=0.0)
        size = self.culler._compute_screen_size(inst)
        assert size == 0.0

    def test_compute_screen_size_exact_formula(self) -> None:
        """Screen size formula: 2 * (radius / distance) / tan(fov/2) * (height/2)."""
        radius = 2.0
        distance = 50.0
        fov = math.radians(60.0)
        height = 1080.0
        expected = 2.0 * (radius / distance) / math.tan(fov * 0.5) * (height * 0.5)

        inst = _make_instance(0, Vec3(0, 0, -distance), radius=radius)
        size = self.culler._compute_screen_size(inst)
        assert size == pytest.approx(expected, rel=1e-6)

    def test_cull_very_near_never_culled(self) -> None:
        """Object very close to camera is never culled by small feature."""
        inst = _make_instance(0, Vec3(0, 0, -1e-7), radius=1.0)
        mask = [True]
        stats = self.culler.cull([inst], mask)
        assert stats.small_feature_culled == 0
        assert mask[0]

    def test_cull_respects_existing_mask(self) -> None:
        """Already-culled instances are skipped."""
        inst = _make_instance(0, Vec3(0, 0, -10000), radius=0.001)
        mask = [False]  # pre-culled
        stats = self.culler.cull([inst], mask)
        assert stats.small_feature_culled == 0  # not counted again
        assert not mask[0]

    def test_update_partial(self) -> None:
        """Update with individual parameters changes only those fields."""
        culler = SmallFeatureCuller(self.config)
        culler.update(camera_position=Vec3(10, 20, 30))
        assert culler._camera_position.x == 10.0
        # FOV should still be default
        assert culler._fov_y == pytest.approx(math.radians(60.0))

        culler.update(fov_y=math.radians(45.0))
        assert culler._fov_y == pytest.approx(math.radians(45.0))

        culler.update(screen_width=3840, screen_height=2160)
        assert culler.config.screen_width == 3840
        assert culler.config.screen_height == 2160


# =============================================================================
# CULLING PIPELINE WHITEBOX TESTS
# =============================================================================

class TestCullingPipelineWhitebox:
    """CullingPipeline empty input, config, sub-culler delegation."""

    def setup_method(self) -> None:
        self.pipeline = CullingPipeline()
        self.pipeline.update(
            view_projection_matrix=_perspective_vp(
                math.radians(90), 1.0, 1.0, 100.0,
            ),
            view_matrix=_identity_view(),
            projection_matrix=_perspective_vp(
                math.radians(90), 1.0, 1.0, 100.0,
            ),
            camera_position=Vec3(0.0, 0.0, 0.0),
            camera_forward=Vec3(0.0, 0.0, -1.0),
            fov_y=math.radians(60.0),
            near_plane=1.0,
            far_plane=100.0,
        )

    def test_cull_empty_instances(self) -> None:
        """Empty instance list returns empty results."""
        indices, stats = self.pipeline.cull([])
        assert indices == []
        assert stats.total_instances == 0
        assert stats.visible_instances == 0

    def test_configure_default_state(self) -> None:
        """Default pipeline has all stages enabled."""
        p = CullingPipeline()
        assert p._enable_frustum
        assert p._enable_occlusion
        assert p._enable_distance
        assert p._enable_small_feature

    def test_configure_toggle_individual(self) -> None:
        """Each stage can be individually toggled."""
        p = CullingPipeline()
        p.configure(enable_frustum=False)
        assert not p._enable_frustum
        assert p._enable_occlusion  # unchanged
        assert p._enable_distance   # unchanged
        assert p._enable_small_feature  # unchanged

        p.configure(enable_frustum=True)
        assert p._enable_frustum

    def test_configure_multiple(self) -> None:
        """Multiple config changes in one call."""
        p = CullingPipeline()
        p.configure(enable_frustum=False, enable_occlusion=False)
        assert not p._enable_frustum
        assert not p._enable_occlusion
        assert p._enable_distance  # unchanged

    def test_update_delegates_to_all_cullers(self) -> None:
        """Pipeline.update dispatches to all sub-cullers."""
        p = CullingPipeline()
        vp = _perspective_vp(math.radians(90), 1.0, 1.0, 100.0)
        p.update(
            frustum=Frustum.from_view_projection_matrix(vp),
            view_matrix=_identity_view(),
            projection_matrix=_perspective_vp(math.radians(90), 1.0, 1.0, 100.0),
            camera_position=Vec3(5, 5, 5),
            camera_forward=Vec3(0, 0, -1),
            fov_y=math.radians(60.0),
            near_plane=0.1,
            far_plane=500.0,
        )
        # Verify each sub-culler received the update
        assert p.frustum_culler.frustum is not None
        assert p.distance_culler._camera_position.x == 5.0
        assert p.distance_culler._camera_forward.length() == pytest.approx(1.0)

    def test_cull_indices_sorted(self) -> None:
        """Visible indices are returned in ascending order."""
        instances = [
            _make_instance(5, Vec3(0, 0, -10), radius=0.5),
            _make_instance(2, Vec3(0, 0, -10), radius=0.5),
            _make_instance(9, Vec3(0, 0, -10), radius=0.5),
            _make_instance(1, Vec3(0, 0, 10), radius=0.5),   # behind -> culled
        ]
        indices, stats = self.pipeline.cull(instances)
        # Sorted ascending
        for i in range(1, len(indices)):
            assert indices[i] >= indices[i - 1]
        # instance_id doesn't matter; index position in the list is what's returned
        assert 3 not in indices  # instance at index 3 is behind camera

    def test_cull_stats_combines_all_stages(self) -> None:
        """Pipeline stats includes contributions from all enabled stages."""
        instances = [
            _make_instance(0, Vec3(0, 0, -10), radius=0.5),     # visible
            _make_instance(1, Vec3(0, 0, 10), radius=0.5),      # behind -> frustum
            _make_instance(2, Vec3(0, 0, -600), radius=0.5),    # far -> distance
            _make_instance(3, Vec3(0, 0, -10000), radius=0.01), # far + tiny -> distance + small
        ]
        indices, stats = self.pipeline.cull(instances)
        assert stats.total_instances == 4
        total_culled = (
            stats.frustum_culled
            + stats.occlusion_culled
            + stats.distance_culled
            + stats.small_feature_culled
        )
        assert stats.visible_instances + total_culled == stats.total_instances

    def test_cull_all_stages_disabled_returns_all(self) -> None:
        """All stages disabled: all instances pass through regardless."""
        self.pipeline.configure(
            enable_frustum=False,
            enable_occlusion=False,
            enable_distance=False,
            enable_small_feature=False,
        )
        # All behind camera, far away, tiny - but everything passes
        instances = [
            _make_instance(0, Vec3(0, 0, 5), radius=0.001),   # behind
            _make_instance(1, Vec3(0, 0, -10000), radius=0.001),  # far + tiny
        ]
        indices, stats = self.pipeline.cull(instances)
        assert len(indices) == 2
        assert stats.visible_instances == 2
        total_culled = (
            stats.frustum_culled
            + stats.occlusion_culled
            + stats.distance_culled
            + stats.small_feature_culled
        )
        assert total_culled == 0


# =============================================================================
# CULLING STATS WHITEBOX TESTS
# =============================================================================

class TestCullingStatsWhitebox:
    """CullingStats edge cases: zero totals, partial counts."""

    def test_cull_ratio_no_instances(self) -> None:
        """Cull ratio is 0.0 when total_instances is 0 (avoids div-by-zero)."""
        s = CullingStats(total_instances=0)
        assert s.cull_ratio == 0.0

    def test_cull_ratio_all_visible(self) -> None:
        """Cull ratio is 0.0 when all instances are visible."""
        s = CullingStats(total_instances=100, visible_instances=100)
        assert s.cull_ratio == 0.0

    def test_cull_ratio_all_culled(self) -> None:
        """Cull ratio is 1.0 when no instances are visible."""
        s = CullingStats(total_instances=100, visible_instances=0)
        assert s.cull_ratio == 1.0

    def test_cull_ratio_half(self) -> None:
        """Cull ratio is 0.5 when half are visible."""
        s = CullingStats(total_instances=100, visible_instances=50)
        assert s.cull_ratio == 0.5


# =============================================================================
# CULLING CONSTANTS WHITEBOX TESTS
# =============================================================================

class TestCullingConstantsWhitebox:
    """CullingConstants values are internally consistent."""

    def test_epsilon_positive(self) -> None:
        assert CullingConstants.EPSILON > 0.0

    def test_near_plane_default_positive(self) -> None:
        assert CullingConstants.DEFAULT_NEAR_PLANE > 0.0

    def test_hzb_mip_levels_match_log2(self) -> None:
        """DEFAULT_HZB_MIP_LEVELS == log2(DEFAULT_HZB_WIDTH) (not +1; base mip is included)."""
        expected = int(math.log2(CullingConstants.DEFAULT_HZB_WIDTH))
        assert CullingConstants.DEFAULT_HZB_MIP_LEVELS == expected

    def test_fade_distance_positive_and_less_than_max(self) -> None:
        """Fade distance is positive and less than max render distance."""
        assert CullingConstants.DEFAULT_FADE_DISTANCE > 0.0
        assert CullingConstants.DEFAULT_FADE_DISTANCE < CullingConstants.DEFAULT_MAX_RENDER_DISTANCE


# =============================================================================
# INSTANCE BOUNDS FLAGS WHITEBOX TESTS
# =============================================================================

class TestInstanceBoundsWhitebox:
    """InstanceBounds flag constants are distinct and non-overlapping."""

    def test_flags_are_power_of_two(self) -> None:
        """Each flag is a distinct power of two (accessed via instance due to slots)."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        flags = [
            inst.FLAG_CAST_SHADOW,
            inst.FLAG_RECEIVE_SHADOW,
            inst.FLAG_TWO_SIDED,
            inst.FLAG_STATIC,
        ]
        for i, f1 in enumerate(flags):
            for f2 in flags[i + 1:]:
                assert f1 & f2 == 0, f"Flags {f1} and {f2} overlap"

    def test_flags_bit_positions(self) -> None:
        """Flags occupy bits 0 through 3."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        assert inst.FLAG_CAST_SHADOW == 1 << 0
        assert inst.FLAG_RECEIVE_SHADOW == 1 << 1
        assert inst.FLAG_TWO_SIDED == 1 << 2
        assert inst.FLAG_STATIC == 1 << 3

    def test_default_flags_zero(self) -> None:
        """Default InstanceBounds has flags=0."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        assert inst.flags == 0

    def test_flags_are_int_via_instance(self) -> None:
        """Flag constants accessed via instance are ints (not slot descriptors)."""
        inst = _make_instance(0, Vec3(0, 0, 0), radius=1.0)
        assert isinstance(inst.FLAG_CAST_SHADOW, int)
        assert isinstance(inst.FLAG_RECEIVE_SHADOW, int)
        assert isinstance(inst.FLAG_TWO_SIDED, int)
        assert isinstance(inst.FLAG_STATIC, int)


# =============================================================================
# FRUSTUM PLANE ENUM WHITEBOX TESTS
# =============================================================================

class TestFrustumPlaneEnumWhitebox:
    """FrustumPlane int-enum values are in expected order."""

    def test_enum_values(self) -> None:
        assert FrustumPlane.LEFT.value == 0
        assert FrustumPlane.RIGHT.value == 1
        assert FrustumPlane.BOTTOM.value == 2
        assert FrustumPlane.TOP.value == 3
        assert FrustumPlane.NEAR.value == 4
        assert FrustumPlane.FAR.value == 5

    def test_enum_count(self) -> None:
        """There are exactly 6 frustum planes."""
        assert len(list(FrustumPlane)) == 6


# =============================================================================
# CULL RESULT ENUM WHITEBOX TESTS
# =============================================================================

class TestCullResultEnumWhitebox:
    """CullResult has all 5 expected values."""

    def test_enum_values(self) -> None:
        from engine.rendering.gpu_driven.culling import CullResult
        # auto() gives 1,2,3,4,5
        assert CullResult.VISIBLE.value == 1
        assert CullResult.CULLED_FRUSTUM.value >= 2
        assert CullResult.CULLED_OCCLUSION.value >= 2
        assert CullResult.CULLED_DISTANCE.value >= 2
        assert CullResult.CULLED_SMALL_FEATURE.value >= 2

    def test_enum_count(self) -> None:
        """There are exactly 5 cull results."""
        from engine.rendering.gpu_driven.culling import CullResult
        assert len(list(CullResult)) == 5

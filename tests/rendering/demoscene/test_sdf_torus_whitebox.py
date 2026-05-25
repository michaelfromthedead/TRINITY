"""
Whitebox tests for sdTorus WGSL function (T-DEMO-1.3).

Tests a Python model of the WGSL implementation, verifying:
  - Formula decomposition matches IQ torus SDF
  - Exact torus surface points yield distance ~0
  - Points inside/outside the torus tube produce correct signed distances
  - Degenerate cases (t.x=0 -> sphere, t.y=0 -> ring)
  - Rotational symmetry around y-axis
  - Negative radii are handled via abs(t)

WGSL implementation (4531cf78):
    fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
        let safe_t = abs(t);
        let q = vec2<f32>(length(p.xz) - safe_t.x, p.y);
        return length(q) - safe_t.y;
    }

IQ reference formula: length(vec2(length(p.xz)-t.x, p.y)) - t.y

WHITEBOX coverage plan:
  Path A:  horizontal_dist = length(p.xz) - t.x  (ring distance decomposition)
  Path B:  full IQ formula: length(vec2(horizontal_dist, p.y)) - t.y
  Path C:  surface point (t.x+t.y, 0, 0) -> distance ~0
  Path D:  major radius point (t.x, 0, 0) -> inside, negative distance
  Path E:  origin (0,0,0) -> inside hole, positive distance ~t.x - t.y
  Path F:  degenerate sphere (t.x=0) -> length(p) - t.y
  Path G:  degenerate ring (t.y=0) -> distance to thin ring
  Path H:  rotational symmetry about y-axis
  Path I:  far-point asymptotic distance
  Path J:  negative radii via abs(t)
"""

import math

import pytest

# =============================================================================
# Python model matching WGSL semantics
# =============================================================================


def py_sd_torus(p, t):
    """Model of WGSL sdTorus: signed distance to torus.

    Formula: length(vec2(length(p.xz) - t.x, p.y)) - t.y
    """
    safe_tx = abs(t[0])
    safe_ty = abs(t[1])
    qx = math.sqrt(p[0] * p[0] + p[2] * p[2]) - safe_tx
    qy = p[1]
    return math.sqrt(qx * qx + qy * qy) - safe_ty


TOL = 1e-12
TOL_SURFACE = 1e-10


# =============================================================================
# Path A: Horizontal ring distance decomposition
# =============================================================================


class TestHorizontalRingDistance:
    """Tests the sub-expression `length(p.xz) - t.x` in isolation.

    This is the horizontal distance from the query point's projection onto
    the xz-plane to the ring of radius t.x. When p.y = 0, the full SDF
    reduces to `abs(length(p.xz) - t.x) - t.y`, so this sub-expression
    determines whether the point is inside or outside the ring.
    """

    def test_on_ring_creates_zero_horizontal(self):
        """When p = (t.x, 0, 0) the horizontal distance is exactly 0."""
        t = (4.0, 1.0)
        px = t[0]
        horizontal = math.sqrt(px * px + 0.0 * 0.0) - t[0]
        assert horizontal == pytest.approx(0.0, abs=TOL), (
            f"Expected horizontal=0 at p=({px},0,0), got {horizontal}"
        )

    def test_inside_ring_negative_horizontal(self):
        """When |p.xz| < t.x, horizontal distance is negative (inside ring)."""
        t = (4.0, 1.0)
        px = 2.0
        horizontal = math.sqrt(px * px + 0.0 * 0.0) - t[0]
        assert horizontal < 0.0, (
            f"Expected negative horizontal inside ring, got {horizontal}"
        )

    def test_outside_ring_positive_horizontal(self):
        """When |p.xz| > t.x, horizontal distance is positive (outside ring)."""
        t = (4.0, 1.0)
        px = 6.0
        horizontal = math.sqrt(px * px + 0.0 * 0.0) - t[0]
        assert horizontal > 0.0, (
            f"Expected positive horizontal outside ring, got {horizontal}"
        )

    def test_horizontal_equality_on_xz_plane(self):
        """Horizontal distance is same for (x,0,z) and (z,0,x).

        This tests that horizontal depends on length(p.xz), not on the
        individual components.
        """
        t = (4.0, 1.0)
        h1 = math.sqrt(3.0 * 3.0 + 4.0 * 4.0) - t[0]
        h2 = math.sqrt(4.0 * 4.0 + 3.0 * 3.0) - t[0]
        assert h1 == pytest.approx(h2, abs=TOL)

    def test_horizontal_scales_with_norm(self):
        """Horizontal distance scales linearly with |p.xz| when far from ring.

        For a point very far out, length(p.xz) >> t.x, so the horizontal
        distance approximates length(p.xz) - t.x.
        """
        t = (4.0, 1.0)
        far = 1000.0
        horizontal = math.sqrt(far * far + far * far) - t[0]
        expected_approx = math.sqrt(2.0) * far - t[0]
        # Relative error should be tiny (dominated by -t.x term which is O(1))
        assert horizontal == pytest.approx(expected_approx, rel=1e-10), (
            f"Horizontal at far point: expected ~{expected_approx}, got {horizontal}"
        )


# =============================================================================
# Path B: Full IQ formula
# =============================================================================


class TestFullFormula:
    """Tests the complete IQ torus SDF: length(vec2(h, p.y)) - t.y.

    The full formula computes the 2D distance in the (horizontal, p.y) plane
    and subtracts the tube radius t.y.
    """

    def test_standard_parameters_positive(self):
        """A point clearly outside the torus produces positive distance."""
        d = py_sd_torus((10.0, 5.0, 0.0), (3.0, 1.0))
        assert d > 0.0, f"Expected positive distance outside torus, got {d}"

    def test_standard_parameters_negative(self):
        """A point clearly inside the torus tube produces negative distance."""
        d = py_sd_torus((3.5, 0.0, 0.0), (3.0, 1.0))
        assert d < 0.0, f"Expected negative distance inside tube, got {d}"

    def test_vertical_displacement_outside(self):
        """A point far above the torus produces positive distance."""
        d = py_sd_torus((3.0, 10.0, 0.0), (3.0, 1.0))
        assert d > 0.0, (
            f"Expected positive distance for far vertical point, got {d}"
        )

    def test_precise_distance_at_known_offset(self):
        """Known offset computation matches formula precisely."""
        t = (3.0, 1.0)
        p = (5.0, 2.0, 0.0)
        # length(p.xz) = 5, horizontal = 5 - 3 = 2
        # q = (2, 2), length(q) = sqrt(8) = 2*sqrt(2) = 2.82842712474619...
        # sd = 2.82842712474619 - 1 = 1.82842712474619
        expected = math.sqrt(8.0) - t[1]
        d = py_sd_torus(p, t)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Expected precise distance {expected}, got {d}"
        )

    def test_under_the_tube_from_below(self):
        """A point below the torus tube has positive distance."""
        d = py_sd_torus((3.0, -2.0, 0.0), (3.0, 1.0))
        assert d > 0.0, (
            f"Expected positive distance below torus, got {d}"
        )


# =============================================================================
# Path C: Exact torus surface points
# =============================================================================


class TestExactTorusSurface:
    """Points on the torus surface should yield a signed distance of ~0.

    The torus surface consists of all points at distance t.y from the ring
    of radius t.x. A point directly on the surface along the x-axis is
    (t.x + t.y, 0, 0).
    """

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (1.0, 0.5),
            (5.0, 2.0),
            (0.5, 0.25),
            (10.0, 0.1),
        ],
    )
    def test_surface_point_positive_x(self, t_major, t_minor):
        """Surface point (t.x + t.y, 0, 0) should have distance ~0."""
        t = (t_major, t_minor)
        p = (t_major + t_minor, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with t={t} should have d=0, got {d}"
        )

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (1.0, 0.5),
            (5.0, 2.0),
        ],
    )
    def test_surface_point_negative_x(self, t_major, t_minor):
        """Surface point (-(t.x + t.y), 0, 0) should have distance ~0."""
        t = (t_major, t_minor)
        p = (-(t_major + t_minor), 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with t={t} should have d=0, got {d}"
        )

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (1.0, 0.5),
        ],
    )
    def test_surface_point_positive_z(self, t_major, t_minor):
        """Surface point (0, 0, t.x + t.y) should have distance ~0."""
        t = (t_major, t_minor)
        p = (0.0, 0.0, t_major + t_minor)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with t={t} should have d=0, got {d}"
        )

    def test_surface_point_vertical_top(self):
        """Point directly above the ring: (t.x, t.y, 0) -> distance ~0."""
        t = (3.0, 1.0)
        p = (t[0], t[1], 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with t={t} should have d=0, got {d}"
        )

    def test_surface_point_vertical_bottom(self):
        """Point directly below the ring: (t.x, -t.y, 0) -> distance ~0."""
        t = (3.0, 1.0)
        p = (t[0], -t[1], 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with t={t} should have d=0, got {d}"
        )


# =============================================================================
# Path D: On major radius (inside torus tube)
# =============================================================================


class TestInsideTube:
    """Points on the major radius circle are inside the torus tube.

    The major radius circle passes through the center of the tube.
    Points on this circle (e.g., (t.x, 0, 0)) are inside the torus
    by exactly t.y, so the signed distance should be -t.y.
    """

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (1.0, 0.5),
            (5.0, 2.0),
            (0.5, 0.25),
            (10.0, 0.1),
        ],
    )
    def test_on_major_radius_negative_tube_distance(self, t_major, t_minor):
        """Point on major radius circle has distance -t.y (inside)."""
        t = (t_major, t_minor)
        p = (t_major, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(-t_minor, abs=TOL), (
            f"Point {p} on major radius with t={t} should have d=-{t_minor}, "
            f"got {d}"
        )

    def test_midway_inside_tube(self):
        """Point midway between ring center and surface: (t.x, t.y/2, 0)."""
        t = (3.0, 1.0)
        p = (t[0], t[1] * 0.5, 0.0)
        d = py_sd_torus(p, t)
        # At (3, 0.5, 0): q = (0, 0.5), length(q)=0.5, sd = 0.5 - 1 = -0.5
        assert d == pytest.approx(-0.5, abs=TOL), (
            f"Expected d=-0.5 at {p}, got {d}"
        )

    def test_negative_along_major_radius(self):
        """On major radius in negative x direction: (-t.x, 0, 0)."""
        t = (4.0, 2.0)
        p = (-t[0], 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(-t[1], abs=TOL), (
            f"Point {p} on major radius should have d=-{t[1]}, got {d}"
        )


# =============================================================================
# Path E: Inside hole (origin)
# =============================================================================


class TestInsideHole:
    """The origin (0,0,0) is inside the hole of the torus.

    At the origin, length(p.xz) = 0 and p.y = 0, so q = (-t.x, 0).
    The distance is | -t.x | - t.y = t.x - t.y, which is positive
    when t.x > t.y (the typical case).
    """

    @pytest.mark.parametrize(
        "t_major, t_minor, expected",
        [
            (3.0, 1.0, 2.0),    # 3 - 1 = 2
            (5.0, 2.0, 3.0),    # 5 - 2 = 3
            (1.0, 0.5, 0.5),    # 1 - 0.5 = 0.5
            (10.0, 0.1, 9.9),   # 10 - 0.1 = 9.9
            (0.5, 0.25, 0.25),  # 0.5 - 0.25 = 0.25
        ],
    )
    def test_origin_inside_hole(self, t_major, t_minor, expected):
        """Origin (0,0,0) has distance t.x - t.y (positive inside hole)."""
        t = (t_major, t_minor)
        d = py_sd_torus((0.0, 0.0, 0.0), t)
        assert d == pytest.approx(expected, abs=TOL), (
            f"At origin with t={t}, expected d={expected}, got {d}"
        )

    def test_origin_when_major_less_than_minor(self):
        """When t.x < t.y, origin is inside tube (negative distance)."""
        t = (1.0, 3.0)
        d = py_sd_torus((0.0, 0.0, 0.0), t)
        expected = t[0] - t[1]  # 1 - 3 = -2
        assert d == pytest.approx(expected, abs=TOL), (
            f"At origin with t={t}, expected d={expected}, got {d}"
        )

    def test_origin_when_major_equals_minor(self):
        """When t.x == t.y, origin is exactly on inner surface (d=0)."""
        t = (2.0, 2.0)
        d = py_sd_torus((0.0, 0.0, 0.0), t)
        assert d == pytest.approx(0.0, abs=TOL), (
            f"At origin with t={t}, expected d=0, got {d}"
        )

    def test_near_origin_positive_distance(self):
        """Points near origin inside the hole remain positive."""
        t = (4.0, 1.0)
        for dx, dz in [(0.1, 0.0), (0.0, 0.1), (0.5, 0.5), (-0.3, 0.4)]:
            d = py_sd_torus((dx, 0.0, dz), t)
            assert d > 0.0, (
                f"Expected positive distance near origin at ({dx},0,{dz}), got {d}"
            )


# =============================================================================
# Path F: Degenerate sphere (t.x = 0)
# =============================================================================


class TestDegenerateSphere:
    """When t.x = 0, the formula should reduce to length(p) - t.y.

    With t.x = 0, the major radius is zero, so the torus degenerates to
    a sphere of radius t.y at the origin:
      sdTorus(p, (0, t.y)) = length(vec2(length(p.xz), p.y)) - t.y
                           = length(p) - t.y
    """

    @pytest.mark.parametrize(
        "p, minor",
        [
            ((0.0, 0.0, 0.0), 1.0),
            ((1.0, 0.0, 0.0), 1.0),
            ((0.0, 2.0, 0.0), 1.0),
            ((3.0, 4.0, 0.0), 5.0),
            ((-2.0, 0.0, 3.0), 4.0),
            ((1.0, 2.0, 3.0), 0.5),
        ],
    )
    def test_sphere_equivalent(self, p, minor):
        """When t.x=0, torus SDF equals sphere SDF: length(p) - t.y."""
        t = (0.0, minor)
        sphere_d = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - minor
        torus_d = py_sd_torus(p, t)
        assert torus_d == pytest.approx(sphere_d, abs=TOL), (
            f"With t.x=0 at {p}, sphere={sphere_d}, torus={torus_d}"
        )

    def test_sphere_surface(self):
        """Sphere surface point: length(p) = t.y -> distance ~0."""
        t = (0.0, 2.5)
        p = (2.5, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Sphere surface point {p} should have d=0, got {d}"
        )

    def test_sphere_inside(self):
        """Point inside sphere: distance negative."""
        t = (0.0, 2.0)
        p = (0.5, 0.0, 0.0)
        d = py_sd_torus(p, t)
        expected = 0.5 - 2.0  # length(p) - t.y = 0.5 - 2.0 = -1.5
        assert d == pytest.approx(expected, abs=TOL), (
            f"Inside sphere at {p}, expected {expected}, got {d}"
        )


# =============================================================================
# Path G: Degenerate ring (t.y = 0)
# =============================================================================


class TestDegenerateRing:
    """When t.y = 0, the torus degenerates to an infinitely thin ring.

    The formula becomes:
      sdTorus(p, (t.x, 0)) = length(vec2(length(p.xz) - t.x, p.y))
    This is the distance to a circle of radius t.x in the xz-plane.
    """

    @pytest.mark.parametrize(
        "p, major",
        [
            ((0.0, 0.0, 0.0), 3.0),
            ((3.0, 0.0, 0.0), 3.0),
            ((4.0, 0.0, 0.0), 3.0),
            ((0.0, 5.0, 0.0), 3.0),
        ],
    )
    def test_ring_distance(self, p, major):
        """When t.y=0, distance is the 2D distance in (h, p.y) space."""
        t = (major, 0.0)
        d = py_sd_torus(p, t)
        expected = math.sqrt(
            (math.sqrt(p[0]**2 + p[2]**2) - major)**2 + p[1]**2
        )
        assert d == pytest.approx(expected, abs=TOL), (
            f"Ring distance at {p} with major={major}: expected {expected}, got {d}"
        )

    def test_ring_on_surface(self):
        """On the ring: (t.x, 0, 0) with t.y=0 -> distance ~0."""
        t = (3.0, 0.0)
        p = (3.0, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"On ring at {p} with t={t}, expected d=0, got {d}"
        )

    def test_ring_above_vertical(self):
        """Directly above the ring: (t.x, h, 0) -> distance = |h|."""
        t = (3.0, 0.0)
        h = 4.0
        p = (t[0], h, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(h, abs=TOL), (
            f"Above ring at {p}, expected d={h}, got {d}"
        )


# =============================================================================
# Path H: Rotational symmetry about y-axis
# =============================================================================


class TestRotationalSymmetry:
    """The torus is rotationally symmetric about the y-axis.

    This means the SDF depends only on the radial distance from y-axis
    (length(p.xz)) and p.y, not on the angle theta in the xz-plane.
    """

    @pytest.mark.parametrize(
        "theta1, theta2",
        [
            (0.0, math.pi),
            (0.0, math.pi / 2),
            (math.pi / 4, math.pi * 3 / 4),
            (0.0, -math.pi),
        ],
    )
    def test_xy_symmetry(self, theta1, theta2):
        """sdTorus((r cos t1, 0, r sin t1)) == sdTorus((r cos t2, 0, r sin t2)).

        Points at the same radial distance should produce the same SDF value
        regardless of azimuthal angle.
        """
        t = (3.0, 1.0)
        r = 4.5  # Some radial distance
        p1 = (r * math.cos(theta1), 0.0, r * math.sin(theta1))
        p2 = (r * math.cos(theta2), 0.0, r * math.sin(theta2))
        d1 = py_sd_torus(p1, t)
        d2 = py_sd_torus(p2, t)
        assert d1 == pytest.approx(d2, abs=TOL), (
            f"Rotational symmetry violated: theta1={theta1}, theta2={theta2}, "
            f"d1={d1}, d2={d2}"
        )

    def test_x_and_z_equivalence(self):
        """(x, 0, z) and (z, 0, x) produce the same distance (xz-plane swap)."""
        t = (3.0, 1.0)
        p1 = (5.0, 2.0, 3.0)
        p2 = (3.0, 2.0, 5.0)
        d1 = py_sd_torus(p1, t)
        d2 = py_sd_torus(p2, t)
        assert d1 == pytest.approx(d2, abs=TOL), (
            f"xz-plane swap violation: p1={p1} d1={d1}, p2={p2} d2={d2}"
        )

    def test_negative_x_symmetry(self):
        """(x, y, z) and (-x, y, z) produce the same distance."""
        t = (3.0, 1.0)
        p_pos = (4.0, 2.0, 1.0)
        p_neg = (-4.0, 2.0, 1.0)
        d_pos = py_sd_torus(p_pos, t)
        d_neg = py_sd_torus(p_neg, t)
        assert d_pos == pytest.approx(d_neg, abs=TOL), (
            f"Sign flip asymmetry: d_pos={d_pos}, d_neg={d_neg}"
        )

    def test_full_circle_invariant(self):
        """Points at the same radius but different angles produce same SDF."""
        t = (4.0, 1.5)
        r_xy = 5.0
        y_val = 0.5
        angles = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]
        ref = py_sd_torus((r_xy, y_val, 0.0), t)
        for theta in angles:
            d = py_sd_torus(
                (r_xy * math.cos(theta), y_val, r_xy * math.sin(theta)), t
            )
            assert d == pytest.approx(ref, abs=TOL), (
                f"Rotation invariance broken at theta={theta}: d={d}, ref={ref}"
            )

    def test_y_axis_vertical_only_symmetry(self):
        """Points on y-axis have same distance regardless of axial position.

        Actually for the torus, points on the y-axis ALL have the same
        horizontal distance (since length(p.xz) = 0), and the SDF becomes
        sqrt(t.x^2 + p.y^2) - t.y. So different y gives different values
        (no invariance along y), but left/right on x-axis should match
        negative-x.
        """
        t = (3.0, 1.0)
        # Check symmetry around the axis: (+x, y, z) == (-x, y, z)
        for x in [1.0, 2.0, 3.5, 5.0]:
            for y in [-2.0, 0.0, 2.0]:
                for z in [-1.0, 0.0, 1.0]:
                    d_pos = py_sd_torus((x, y, z), t)
                    d_neg = py_sd_torus((-x, y, z), t)
                    assert d_pos == pytest.approx(d_neg, abs=TOL), (
                        f"Asymmetry at ({x},{y},{z}): d_pos={d_pos}, d_neg={d_neg}"
                    )


# =============================================================================
# Path I: Far-point asymptotic distance
# =============================================================================


class TestFarPointAsymptotic:
    """For points far from the torus, the distance should asymptotically
    approach length(p) - (t.x + t.y) along the x-axis.

    When the query point is far away compared to both radii, the torus
    approximates a point mass at the origin, so the SDF behaves like
    the distance to the origin minus the torus's effective span.
    """

    def test_far_along_x_axis(self):
        """Far point along x-axis: sd = d - t.x - t.y (exact)."""
        t = (3.0, 1.0)
        far = 1000.0
        d = py_sd_torus((far, 0.0, 0.0), t)
        # length(p) = far, horizontal = far - t.x, q = (far - t.x, 0)
        # length(q) = far - t.x, sd = far - t.x - t.y = far - 4
        expected = far - t[0] - t[1]
        assert d == pytest.approx(expected, rel=1e-12), (
            f"Far point asymptotic: expected {expected}, got {d}"
        )

    def test_far_diagonal(self):
        """Far point along diagonal x=z: distance still matches formula."""
        t = (3.0, 1.0)
        far = 1000.0
        d = py_sd_torus((far, 0.0, far), t)
        # length(p.xz) = sqrt(2)*far
        # qx = sqrt(2)*far - t.x, qy = 0
        # length(q) = sqrt(2)*far - t.x
        # sd = sqrt(2)*far - t.x - t.y
        expected = math.sqrt(2.0) * far - t[0] - t[1]
        assert d == pytest.approx(expected, rel=1e-12), (
            f"Far diagonal asymptotic: expected {expected}, got {d}"
        )

    def test_far_vertical(self):
        """Far point along y-axis: sd = sqrt(t.x^2 + far^2) - t.y (exact)."""
        t = (3.0, 1.0)
        far = 1000.0
        d = py_sd_torus((0.0, far, 0.0), t)
        # length(p.xz) = 0, qx = -t.x, qy = far
        # length(q) = sqrt(t.x^2 + far^2), sd = sqrt(t.x^2 + far^2) - t.y
        expected_exact = math.sqrt(t[0]**2 + far**2) - t[1]
        assert d == pytest.approx(expected_exact, abs=TOL), (
            f"Far vertical: expected {expected_exact}, got {d}"
        )

    def test_far_point_asymptotic_ratio(self):
        """SDF/length(p) approaches 1 as p goes to infinity along x-axis."""
        t = (3.0, 1.0)
        for scale in [100.0, 1000.0, 10000.0, 100000.0]:
            p = (scale, 0.0, 0.0)
            d = py_sd_torus(p, t)
            mag = abs(p[0])
            ratio = d / mag
            # Ratio approaches 1 from below as scale -> inf
            assert 0.9 < ratio < 1.0, (
                f"At scale={scale}, ratio d/mag={ratio} not near 1"
            )

    def test_far_point_ordering(self):
        """Two far points: farther one has larger SDF (monotonic)."""
        t = (3.0, 1.0)
        d1 = py_sd_torus((100.0, 0.0, 0.0), t)
        d2 = py_sd_torus((200.0, 0.0, 0.0), t)
        assert d2 > d1, (
            f"Far point ordering violated: d(100)={d1}, d(200)={d2}"
        )


# =============================================================================
# Path J: Negative radii via abs(t)
# =============================================================================


class TestNegativeRadii:
    """The implementation uses abs(t) to handle negative radii.

    This ensures that t.x and t.y are always non-negative in the
    computation, producing the same result as if the user passed
    positive values. This is important because negative radii would
    physically be meaningless but could arise from animation blending.
    """

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (5.0, 2.0),
            (0.5, 0.25),
            (1.0, 0.1),
        ],
    )
    def test_negative_major_radius(self, t_major, t_minor):
        """Negative t.x is equivalent to positive t.x (abs)."""
        t_pos = (t_major, t_minor)
        t_neg = (-t_major, t_minor)
        p = (4.0, 2.0, 1.0)
        d_pos = py_sd_torus(p, t_pos)
        d_neg = py_sd_torus(p, t_neg)
        assert d_pos == pytest.approx(d_neg, abs=TOL), (
            f"Negative t.x mismatch: t_pos={t_pos}, t_neg={t_neg}, "
            f"d_pos={d_pos}, d_neg={d_neg}"
        )

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (5.0, 2.0),
        ],
    )
    def test_negative_minor_radius(self, t_major, t_minor):
        """Negative t.y is equivalent to positive t.y (abs)."""
        t_pos = (t_major, t_minor)
        t_neg = (t_major, -t_minor)
        p = (2.0, 1.0, 3.0)
        d_pos = py_sd_torus(p, t_pos)
        d_neg = py_sd_torus(p, t_neg)
        assert d_pos == pytest.approx(d_neg, abs=TOL), (
            f"Negative t.y mismatch: t_pos={t_pos}, t_neg={t_neg}, "
            f"d_pos={d_pos}, d_neg={d_neg}"
        )

    @pytest.mark.parametrize(
        "t_major, t_minor",
        [
            (3.0, 1.0),
            (5.0, 2.0),
        ],
    )
    def test_both_negative(self, t_major, t_minor):
        """Both t.x and t.y negative are equivalent to both positive."""
        t_pos = (t_major, t_minor)
        t_neg = (-t_major, -t_minor)
        p = (4.0, 2.0, 1.0)
        d_pos = py_sd_torus(p, t_pos)
        d_neg = py_sd_torus(p, t_neg)
        assert d_pos == pytest.approx(d_neg, abs=TOL), (
            f"Both negative mismatch: t_pos={t_pos}, t_neg={t_neg}, "
            f"d_pos={d_pos}, d_neg={d_neg}"
        )

    def test_zero_radii_edge(self):
        """Both radii zero: torus degenerates to a point at origin."""
        d = py_sd_torus((5.0, 0.0, 0.0), (0.0, 0.0))
        assert d == pytest.approx(5.0, abs=TOL), (
            f"Zero radii point: expected 5.0, got {d}"
        )
        d_origin = py_sd_torus((0.0, 0.0, 0.0), (0.0, 0.0))
        assert d_origin == pytest.approx(0.0, abs=TOL), (
            f"Zero radii at origin: expected 0, got {d_origin}"
        )

    def test_negative_major_equivalent_on_surface(self):
        """Surface check: negative t.x still gives correct surface distance."""
        t = (-3.0, 1.0)
        p = (4.0, 0.0, 0.0)  # abs(t.x) + t.y = 4 -> on surface
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} with negative t={t}: expected d=0, got {d}"
        )

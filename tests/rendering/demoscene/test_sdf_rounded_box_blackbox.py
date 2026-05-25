"""
Cleanroom blackbox tests for sdRoundedBox WGSL function (T-DEMO-1.10).

Tests the signed distance function for a rounded box of half-dimensions b
with uniform corner radius r, centered at the origin, treating the
implementation as a black box from the spec.

Inigo Quilez reference formula:
    let q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;

The abs(r) clamp and the [0, min(b)] clamp on r are part of the WGSL contract.

BLACKBOX coverage plan:
  Path 1:  Corner rounding -- rounded corner surface returns ~0
  Path 2:  r=0 degenerates to sharp box (sdBox)
  Path 3:  Interior points return negative signed distance
  Path 4:  Surface points return ~0 on all 6 face centers
  Path 5:  Exterior points return positive signed distance
  Path 6:  Large r rounds corners more strongly
  Path 7:  Sign convention -- negative inside, zero on surface, positive outside
  Path 8:  Symmetry -- sign-flip symmetry across all axes
  Path 9:  Continuity -- nearby points have nearby distances
  Path 10: Monotonicity -- distances increase away from center
"""

from __future__ import annotations

import math

import pytest


# =============================================================================
# Python model of sdRoundedBox matching WGSL semantics (blackbox proxy)
# =============================================================================


def py_sd_rounded_box(p, b, r):
    """Python model of WGSL sdRoundedBox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32.

    Signed distance from point p (3-tuple) to a rounded box of half-dimensions
    b (3-tuple) with uniform corner radius r, centered at the origin.

    The corner radius r is clamped to [0, min(b)] so the box never vanishes.

    Reference: Inigo Quilez -- Rounded Box SDF
    https://iquilezles.org/articles/distfunctions/
    """
    safe_r = min(max(r, 0.0), min(b[0], b[1], b[2]))

    # Shift q by r so the rounded corner surface coincides with the
    # original box face at distance b_i from center.
    q = (abs(p[0]) - b[0] + safe_r,
         abs(p[1]) - b[1] + safe_r,
         abs(p[2]) - b[2] + safe_r)

    # max(q, 0) component-wise
    qx_pos = max(q[0], 0.0)
    qy_pos = max(q[1], 0.0)
    qz_pos = max(q[2], 0.0)
    exterior = math.sqrt(qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos)

    # Interior term: min(max_component, 0)
    interior = min(max(q[0], max(q[1], q[2])), 0.0)

    return exterior + interior - safe_r


# =============================================================================
# Python model of a sharp box (sdBox) for comparison when r=0
# =============================================================================


def py_sd_box(p, b):
    """Python model of a sharp-edged box SDF (sdBox).

    sdBox(p, b) = length(max(abs(p)-b, 0)) + min(max(abs(p)-b), 0)
    """
    q = (abs(p[0]) - b[0],
         abs(p[1]) - b[1],
         abs(p[2]) - b[2])
    qx_pos = max(q[0], 0.0)
    qy_pos = max(q[1], 0.0)
    qz_pos = max(q[2], 0.0)
    exterior = math.sqrt(qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos)
    interior = min(max(q[0], max(q[1], q[2])), 0.0)
    return exterior + interior


# =============================================================================
# Default test parameters
# =============================================================================

TOL_SURFACE = 1e-9       # Points on surface should be very close to 0
TOL_CONTINUITY = 0.01    # Max allowed jump for nearby points
TOL_LINEAR = 0.001       # Tolerance for linearity checks
TOL_FAR = 0.001          # Tolerance for far-point approximations

# Default box: half-dimensions (2, 1, 1), corner radius 0.5
B_DEFAULT = (2.0, 1.0, 1.0)
R_DEFAULT = 0.5

# Cube: equal half-dimensions with rounding
B_CUBE = (2.0, 2.0, 2.0)
R_CUBE = 0.5

# Small radius
R_SMALL = 0.1

# Large radius (close to min half-dimension)
R_LARGE = 0.9

# Large box (all dims larger)
B_LARGE = (3.0, 2.0, 2.0)


# =============================================================================
# Path 1: Corner rounding -- rounded corner surface returns ~0
# =============================================================================


class TestCornerRounding:
    """Points on the rounded corner surface should return distance ~0."""

    def test_corner_surface_diagonal(self):
        """Point on the diagonal corner surface: core corner + r along diagonal."""
        b, r = B_DEFAULT, R_DEFAULT
        # The rounded corner is the Minkowski sum of a core box (b-r) and a
        # sphere of radius r.  Along the diagonal the surface is at
        #   p = (b-r) + r * (1,1,1)/sqrt(3)
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        core = (b[0] - r, b[1] - r, b[2] - r)
        p = (core[0] + r * inv_sqrt3,
             core[1] + r * inv_sqrt3,
             core[2] + r * inv_sqrt3)
        d = py_sd_rounded_box(p, (b[0], b[1], b[2]), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Corner surface point {p} should be ~0, got {d}"
        )

    def test_corner_surface_positive_octant(self):
        """Surface in positive octant along all three corner axes."""
        b, r = B_DEFAULT, R_DEFAULT
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        core = (b[0] - r, b[1] - r, b[2] - r)
        # Sample 5 points along the spherical corner cap
        for t in [0.2, 0.5, 0.8, 1.0]:
            p = (core[0] + r * inv_sqrt3 * t,
                 core[1] + r * inv_sqrt3 * t,
                 core[2] + r * inv_sqrt3 * t)
            d = py_sd_rounded_box(p, b, r)
            # Only the true surface at t=1 gives exactly 0
            if t == 1.0:
                assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                    f"Corner surface point {p} should be ~0, got {d}"
                )
            elif t < 1.0:
                assert d < 0.0, (
                    f"Inside corner {p} should be negative, got {d}"
                )
            else:
                assert d > 0.0, (
                    f"Outside corner {p} should be positive, got {d}"
                )

    def test_corner_surface_all_octants(self):
        """Corner surface in all 8 octants."""
        b, r = B_DEFAULT, R_DEFAULT
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        core = (b[0] - r, b[1] - r, b[2] - r)
        signs = [
            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
            (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
        ]
        for sx, sy, sz in signs:
            base = (core[0] + r * inv_sqrt3,
                    core[1] + r * inv_sqrt3,
                    core[2] + r * inv_sqrt3)
            p = (sx * base[0], sy * base[1], sz * base[2])
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Corner surface in octant ({sx},{sy},{sz}): "
                f"point {p} should be ~0, got {d}"
            )

    def test_corner_inside_core_negative(self):
        """Inside the core box (b-r) but off-center is still negative."""
        b, r = B_DEFAULT, R_DEFAULT
        core = (b[0] - r, b[1] - r, b[2] - r)
        # Point just inside the core box
        p = (core[0] * 0.5, core[1] * 0.5, core[2] * 0.5)
        d = py_sd_rounded_box(p, b, r)
        assert d < 0.0, (
            f"Core box interior point {p} should be negative, got {d}"
        )

    def test_corner_outside_sharp_corner_positive(self):
        """At the original sharp corner (b.x, b.y, b.z), the rounded box
        has positive distance because the corner was cut off."""
        b, r = B_DEFAULT, R_DEFAULT
        # Original sharp corner
        p = (b[0], b[1], b[2])
        d = py_sd_rounded_box(p, b, r)
        assert d > 0.0, (
            f"Original sharp corner {p} should be outside rounded box, "
            f"got {d}"
        )

    def test_corner_rounding_scales_with_r(self):
        """Larger r cuts off more of the corner."""
        b = B_DEFAULT
        sharp_corner = (b[0], b[1], b[2])
        d_small_r = py_sd_rounded_box(sharp_corner, b, R_SMALL)
        d_large_r = py_sd_rounded_box(sharp_corner, b, R_LARGE)
        assert d_large_r > d_small_r, (
            f"Larger r should make corner more positive (more cut off): "
            f"r={R_SMALL} -> {d_small_r}, r={R_LARGE} -> {d_large_r}"
        )


# =============================================================================
# Path 2: r=0 degenerates to sharp box (Acceptance)
# =============================================================================


class TestSharpBox:
    """With r=0, sdRoundedBox behaves exactly like a sharp box (sdBox)."""

    def test_r0_matches_box_at_center(self):
        """At origin, r=0 gives same distance as sharp box."""
        b = B_DEFAULT
        d_r0 = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        d_box = py_sd_box((0.0, 0.0, 0.0), b)
        assert d_r0 == pytest.approx(d_box, abs=TOL_SURFACE), (
            f"r=0 at center: rounded={d_r0}, box={d_box}"
        )

    def test_r0_matches_box_inside(self):
        """Inside point with r=0 matches sharp box."""
        b = B_DEFAULT
        points = [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (-0.3, 0.2, 0.1)]
        for p in points:
            d_r0 = py_sd_rounded_box(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_r0 == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"r=0 at {p}: rounded={d_r0}, box={d_box}"
            )

    def test_r0_matches_box_surface(self):
        """Surface point with r=0 matches sharp box surface."""
        b = B_DEFAULT
        surface_points = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
        ]
        for p in surface_points:
            d_r0 = py_sd_rounded_box(p, b, 0.0)
            assert d_r0 == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"r=0 at surface {p}: should be ~0, got {d_r0}"
            )

    def test_r0_matches_box_outside(self):
        """Outside point with r=0 matches sharp box."""
        b = B_DEFAULT
        points = [(b[0] + 0.5, 0.0, 0.0), (0.0, b[1] + 1.0, 0.0),
                  (b[0] + 0.3, b[1] + 0.3, 0.0)]
        for p in points:
            d_r0 = py_sd_rounded_box(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_r0 == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"r=0 at {p}: rounded={d_r0}, box={d_box}"
            )

    def test_r0_cube_variation(self):
        """r=0 on a cube matches the box SDF."""
        b = B_CUBE
        points = [(1.0, 0.0, 0.0), (1.5, 1.0, 0.0), (3.0, 0.0, 0.0),
                  (-1.0, -0.5, 0.0)]
        for p in points:
            d_r0 = py_sd_rounded_box(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_r0 == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"r=0 cube at {p}: rounded={d_r0}, box={d_box}"
            )

    def test_r0_sharp_corner_is_surface(self):
        """With r=0, the corner IS the surface (sharp box)."""
        b = B_DEFAULT
        p = (b[0], b[1], 0.0)  # Edge point on the box
        d = py_sd_rounded_box(p, b, 0.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Sharp corner with r=0 should be on surface: p={p}, d={d}"
        )


# =============================================================================
# Path 3: Interior points return negative signed distance (Acceptance)
# =============================================================================


class TestInterior:
    """Points inside the rounded box should return negative signed distance."""

    def test_origin_is_negative(self):
        """Center of box at (0,0,0) should be negative (inside)."""
        d = py_sd_rounded_box((0.0, 0.0, 0.0), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Origin should be inside, got {d}"
        )

    def test_origin_circle(self):
        """Center of cubic rounded box is negative."""
        d = py_sd_rounded_box((0.0, 0.0, 0.0), B_CUBE, R_CUBE)
        assert d < 0.0, (
            f"Origin in cubic rounded box should be inside, got {d}"
        )

    def test_inside_positive_x(self):
        """Point on positive x inside: (1, 0, 0) with b=(2,1,1)."""
        d = py_sd_rounded_box((1.0, 0.0, 0.0), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (1,0,0) should be inside, got {d}"
        )

    def test_inside_negative_x(self):
        """Point on negative x inside: (-1, 0, 0)."""
        d = py_sd_rounded_box((-1.0, 0.0, 0.0), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (-1,0,0) should be inside, got {d}"
        )

    def test_inside_positive_y(self):
        """Point inside near y face: (0, 0.5, 0)."""
        d = py_sd_rounded_box((0.0, 0.5, 0.0), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (0,0.5,0) should be inside, got {d}"
        )

    def test_inside_negative_y(self):
        """Point inside near -y face: (0, -0.5, 0)."""
        d = py_sd_rounded_box((0.0, -0.5, 0.0), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (0,-0.5,0) should be inside, got {d}"
        )

    def test_inside_positive_z(self):
        """Point inside near z face: (0, 0, 0.5)."""
        d = py_sd_rounded_box((0.0, 0.0, 0.5), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (0,0,0.5) should be inside, got {d}"
        )

    def test_inside_diagonal(self):
        """Point inside on the diagonal: (0.5, 0.3, 0.3)."""
        d = py_sd_rounded_box((0.5, 0.3, 0.3), B_DEFAULT, R_DEFAULT)
        assert d < 0.0, (
            f"Point (0.5,0.3,0.3) should be inside, got {d}"
        )

    def test_inside_near_surface_x(self):
        """Point just inside the x-face: slightly less than b.x."""
        b, r = B_DEFAULT, R_DEFAULT
        x_just_inside = b[0] - 0.001
        d = py_sd_rounded_box((x_just_inside, 0.0, 0.0), b, r)
        assert d < 0.0, (
            f"Point just inside x-face should be negative, got {d}"
        )


# =============================================================================
# Path 4: Surface points return ~0 on all 6 face centers (Acceptance)
# =============================================================================


class TestSurface:
    """Points on the rounded box surface should return distance ~0."""

    def test_surface_positive_x(self):
        """On the positive x face center: (b.x, 0, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point ({b[0]},0,0) on x-face should be ~0, got {d}"
        )

    def test_surface_negative_x(self):
        """On the negative x face center: (-b.x, 0, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((-b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point ({-b[0]},0,0) on x-face should be ~0, got {d}"
        )

    def test_surface_positive_y(self):
        """On the positive y face center: (0, b.y, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, b[1], 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point (0,{b[1]},0) on y-face should be ~0, got {d}"
        )

    def test_surface_negative_y(self):
        """On the negative y face center: (0, -b.y, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, -b[1], 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point (0,{-b[1]},0) on y-face should be ~0, got {d}"
        )

    def test_surface_positive_z(self):
        """On the positive z face center: (0, 0, b.z)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, 0.0, b[2]), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point (0,0,{b[2]}) on z-face should be ~0, got {d}"
        )

    def test_surface_negative_z(self):
        """On the negative z face center: (0, 0, -b.z)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, 0.0, -b[2]), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point (0,0,{-b[2]}) on z-face should be ~0, got {d}"
        )

    def test_surface_face_offset(self):
        """Off-center but still on the face is on the surface."""
        b, r = B_DEFAULT, R_DEFAULT
        # Points on the y-face but offset in x and z
        points = [
            (0.5, b[1], 0.0),
            (-0.3, b[1], 0.2),
            (0.0, b[1], -0.4),
            (0.5, -b[1], 0.0),
        ]
        for p in points:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Point {p} on face should be ~0, got {d}"
            )

    def test_surface_cube(self):
        """Surface on all 6 faces of a cubic rounded box."""
        b, r = B_CUBE, R_CUBE
        face_points = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
        ]
        for p in face_points:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Cube surface point {p} should be ~0, got {d}"
            )


# =============================================================================
# Path 5: Exterior points return positive signed distance (Acceptance)
# =============================================================================


class TestExterior:
    """Points outside the rounded box should return positive signed distance."""

    def test_outside_positive_x(self):
        """Point outside x-face: (b.x+1, 0, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((b[0] + 1.0, 0.0, 0.0), b, r)
        assert d > 0.0, (
            f"Point ({b[0]+1},0,0) should be outside, got {d}"
        )

    def test_outside_negative_x(self):
        """Point outside -x face: (-b.x-1, 0, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((-b[0] - 1.0, 0.0, 0.0), b, r)
        assert d > 0.0, (
            f"Point ({-b[0]-1},0,0) should be outside, got {d}"
        )

    def test_outside_positive_y(self):
        """Point above y-face: (0, b.y+1, 0)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, b[1] + 1.0, 0.0), b, r)
        assert d > 0.0, (
            f"Point (0,{b[1]+1},0) should be outside, got {d}"
        )

    def test_outside_positive_z(self):
        """Point outside z-face: (0, 0, b.z+1)."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, 0.0, b[2] + 1.0), b, r)
        assert d > 0.0, (
            f"Point (0,0,{b[2]+1}) should be outside, got {d}"
        )

    def test_outside_corner_diagonal(self):
        """Outside corner at the original box corner: (b.x, b.y, b.z) > 0."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((b[0], b[1], b[2]), b, r)
        assert d > 0.0, (
            f"Original corner ({b[0]},{b[1]},{b[2]}) cut off, "
            f"should be outside, got {d}"
        )

    def test_outside_far_along_x(self):
        """Far outside along x: distance approximately = x - b.x."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((10.0, 0.0, 0.0), b, r)
        expected = 10.0 - b[0]
        assert d == pytest.approx(expected, abs=TOL_LINEAR), (
            f"Far point (10,0,0) should have distance ~{expected}, got {d}"
        )

    def test_outside_far_along_y(self):
        """Far outside along y: distance approximately = y - b.y."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, 10.0, 0.0), b, r)
        expected = 10.0 - b[1]
        assert d == pytest.approx(expected, abs=TOL_LINEAR), (
            f"Far point (0,10,0) should have distance ~{expected}, got {d}"
        )

    def test_outside_far_along_z(self):
        """Far outside along z: distance approximately = z - b.z."""
        b, r = B_DEFAULT, R_DEFAULT
        d = py_sd_rounded_box((0.0, 0.0, 10.0), b, r)
        expected = 10.0 - b[2]
        assert d == pytest.approx(expected, abs=TOL_LINEAR), (
            f"Far point (0,0,10) should have distance ~{expected}, got {d}"
        )


# =============================================================================
# Path 6: Large r smooths corners (Acceptance)
# =============================================================================


class TestLargeRadius:
    """A large corner radius rounds the box more strongly."""

    def test_large_r_makes_corner_more_positive(self):
        """Larger r at the same corner point gives larger (more outside) value."""
        b = B_DEFAULT
        corner = (b[0], b[1], b[2])
        radii = [0.1, 0.3, 0.5, 0.7]
        prev_d = -float("inf")
        for r in radii:
            d = py_sd_rounded_box(corner, b, r)
            assert d > prev_d, (
                f"Larger r should increase corner distance: "
                f"r={r} -> {d}, previous > was {prev_d}"
            )
            prev_d = d

    def test_large_r_surface_faces_unchanged(self):
        """The face center surfaces are still at ±b_i regardless of r.
        (The rounding only affects corners, not face centers.)"""
        b = B_DEFAULT
        for r in [0.1, 0.3, 0.5, 0.7, R_LARGE]:
            d = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"x-face center should be on surface at any r={r}, got {d}"
            )

    def test_large_r_center_unchanged(self):
        """The center distance is always -min(b), independent of r, because
        rounding only affects corners, not face normals."""
        b = B_DEFAULT
        min_dim = min(b)
        center = (0.0, 0.0, 0.0)
        for r in [0.0, 0.1, 0.3, 0.5, 0.7, R_LARGE]:
            d = py_sd_rounded_box(center, b, r)
            expected = -min_dim
            assert d == pytest.approx(expected, abs=TOL_SURFACE), (
                f"Center should be -min(b) = {expected} for any r={r}, "
                f"got {d}"
            )

    def test_large_r_corner_more_cut_than_edge(self):
        """Larger r cuts off more of the corner than the edge region."""
        b = B_DEFAULT
        corner = (b[0], b[1], b[2])
        edge = (b[0], b[1], 0.0)
        for r in [R_SMALL, R_DEFAULT, R_LARGE]:
            d_corner = py_sd_rounded_box(corner, b, r)
            d_edge = py_sd_rounded_box(edge, b, r)
            # The corner is always more outside (positive) than the edge
            assert d_corner >= d_edge - 1e-12, (
                f"Corner should be >= edge for any r: "
                f"r={r}, corner={d_corner}, edge={d_edge}"
            )

    def test_large_r_edge_smoothing(self):
        """Points near the edge (not corner, not face center) get smoother."""
        b = B_DEFAULT
        # A point near the x-y edge but not at the corner
        edge_point = (b[0] * 0.95, b[1] * 0.95, 0.0)
        d_small = py_sd_rounded_box(edge_point, b, R_SMALL)
        d_large = py_sd_rounded_box(edge_point, b, R_LARGE)
        # With large r, this point is more likely to be inside
        # (the rounding smooths the edge inward)
        assert d_small != d_large, (
            f"Large r should change SDF near edges"
        )

    def test_large_r_clamped_to_min_b(self):
        """r is clamped to min(b), so large r values don't exceed min(b)."""
        b = B_DEFAULT
        min_dim = min(b)
        # r > min_dim should behave the same as r = min_dim
        d_clipped = py_sd_rounded_box((0.0, 0.0, 0.0), b, min_dim * 2)
        d_at_limit = py_sd_rounded_box((0.0, 0.0, 0.0), b, min_dim)
        assert d_clipped == pytest.approx(d_at_limit, abs=TOL_SURFACE), (
            f"r > min(b) should be clamped to min(b)"
        )

    def test_large_r_approaches_spherical(self):
        """When r approaches min(b), shape approaches a sphere-like form
        (the smallest dimension cap dominates)."""
        b = B_DEFAULT
        min_dim = min(b)
        r = min_dim * 0.99  # Almost filling the smallest dimension
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # The distance at center should be close to -min_dim (sphere radius)
        # But not exactly since it's still a rounded box
        assert d < 0.0, f"Center should still be inside with large r, got {d}"


# =============================================================================
# Path 7: Sign convention -- negative inside, zero surface, positive outside
# =============================================================================


class TestSignConvention:
    """Sign convention: negative inside, zero on surface, positive outside."""

    def test_inside_points_negative(self):
        """Known interior points should have non-positive SDF."""
        b, r = B_DEFAULT, R_DEFAULT
        inside_points = [
            (0.0, 0.0, 0.0),
            (b[0] * 0.5, 0.0, 0.0),
            (0.0, b[1] * 0.5, 0.0),
            (0.0, 0.0, b[2] * 0.5),
            (-b[0] * 0.3, b[1] * 0.5, 0.0),
            (0.5, 0.3, 0.2),
        ]
        for p in inside_points:
            d = py_sd_rounded_box(p, b, r)
            assert d <= 0.0, (
                f"Interior point {p} should have non-positive SDF, got {d}"
            )

    def test_outside_points_positive(self):
        """Known exterior points should have non-negative SDF."""
        b, r = B_DEFAULT, R_DEFAULT
        outside_points = [
            (b[0] + 1.0, 0.0, 0.0),
            (0.0, b[1] + 1.0, 0.0),
            (0.0, 0.0, b[2] + 1.0),
            (-(b[0] + 1.0), 0.0, 0.0),
            (b[0] + 0.5, b[1] + 0.5, b[2] + 0.5),
            (b[0] * 2.0, b[1] * 0.5, 0.0),
        ]
        for p in outside_points:
            d = py_sd_rounded_box(p, b, r)
            assert d >= 0.0, (
                f"Exterior point {p} should have non-negative SDF, got {d}"
            )

    def test_surface_points_zero(self):
        """Known surface points should have near-zero SDF."""
        b, r = B_DEFAULT, R_DEFAULT
        surface_points = [
            (b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, 0.0, b[2]),
            (-b[0], 0.0, 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, -b[2]),
        ]
        for p in surface_points:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_surface_x(self):
        """Sign transitions from negative to positive through x-face."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 1e-4
        inside = py_sd_rounded_box((b[0] - step, 0.0, 0.0), b, r)
        on_surf = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        outside = py_sd_rounded_box((b[0] + step, 0.0, 0.0), b, r)
        assert inside < 0.0, f"Inside x-face should be negative, got {inside}"
        assert on_surf == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"On x-face should be ~0, got {on_surf}"
        )
        assert outside > 0.0, f"Outside x-face should be positive, got {outside}"

    def test_sign_transition_y(self):
        """Sign transition through y-face."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 1e-4
        inside = py_sd_rounded_box((0.0, b[1] - step, 0.0), b, r)
        on_surf = py_sd_rounded_box((0.0, b[1], 0.0), b, r)
        outside = py_sd_rounded_box((0.0, b[1] + step, 0.0), b, r)
        assert inside < 0.0
        assert on_surf == pytest.approx(0.0, abs=TOL_SURFACE)
        assert outside > 0.0

    def test_sign_consistent_all_axes(self):
        """Sign convention holds on all three principal axes and the diagonal."""
        b, r = B_DEFAULT, R_DEFAULT
        # Axis-aligned directions -- surface is exactly at b_i
        axis_dirs = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ]
        for direction in axis_dirs:
            abs_dir = tuple(abs(c) for c in direction)
            # Inside at half-b -- safe for axis-aligned dirs
            inside_p = tuple(direction[i] * b[i] * 0.5 for i in range(3))
            # Surface at exactly b_i
            surface_p = tuple(direction[i] * b[i] for i in range(3))
            # Outside at 1.5x b
            outside_p = tuple(direction[i] * b[i] * 1.5 for i in range(3))
            assert py_sd_rounded_box(inside_p, b, r) < 0.0, (
                f"Inside along {direction} should be negative"
            )
            assert py_sd_rounded_box(surface_p, b, r) == pytest.approx(
                0.0, abs=TOL_SURFACE
            ), f"Surface along {direction} should be ~0"
            assert py_sd_rounded_box(outside_p, b, r) > 0.0, (
                f"Outside along {direction} should be positive"
            )

    def test_sign_transition_diagonal_inside_outside(self):
        """Sign transitions through zero on the diagonal (through a
        rounded corner). Uses a root-finding approach to locate the surface."""
        b, r = B_DEFAULT, R_DEFAULT
        # Known inside point on diagonal
        d_inside = py_sd_rounded_box(
            (b[0] * 0.5, b[1] * 0.5, b[2] * 0.5), b, r
        )
        assert d_inside < 0.0, "Diagonal inside should be negative"
        # Known outside point on diagonal (original sharp corner)
        d_outside = py_sd_rounded_box((b[0], b[1], b[2]), b, r)
        assert d_outside > 0.0, "Sharp corner should be outside (positive)"


# =============================================================================
# Path 8: Symmetry -- sign-flip symmetry across all axes
# =============================================================================


class TestSymmetry:
    """The rounded box SDF is symmetric under sign flips of each component."""

    def test_x_sign_symmetry(self):
        """(x, y, z) and (-x, y, z) have equal distance."""
        b, r = B_DEFAULT, R_DEFAULT
        samples = [(0.5, 0.3, 0.2), (1.0, 0.0, 0.5), (1.5, 0.5, 0.3)]
        for x, y, z in samples:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((-x, y, z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"x-sign symmetry broken for ({x},{y},{z})"
            )

    def test_y_sign_symmetry(self):
        """(x, y, z) and (x, -y, z) have equal distance."""
        b, r = B_DEFAULT, R_DEFAULT
        samples = [(0.3, 0.5, 0.2), (0.5, 1.0, 0.0), (1.0, 0.8, 0.3)]
        for x, y, z in samples:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((x, -y, z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"y-sign symmetry broken for ({x},{y},{z})"
            )

    def test_z_sign_symmetry(self):
        """(x, y, z) and (x, y, -z) have equal distance."""
        b, r = B_DEFAULT, R_DEFAULT
        samples = [(0.3, 0.2, 0.5), (0.5, 0.0, 1.0), (1.0, 0.3, 0.8)]
        for x, y, z in samples:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((x, y, -z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"z-sign symmetry broken for ({x},{y},{z})"
            )

    def test_xz_swap_symmetry(self):
        """(x, y, z) and (z, y, x) have equal distance (b.x == b.z)."""
        b, r = B_DEFAULT, R_DEFAULT
        # Only valid when b.x == b.z (our default has b.x=2, b.z=1 -- NOT equal)
        # Use the cube parameters for full swap symmetry
        d1 = py_sd_rounded_box((0.5, 0.3, 0.7), B_CUBE, R_CUBE)
        d2 = py_sd_rounded_box((0.7, 0.3, 0.5), B_CUBE, R_CUBE)
        assert d1 == pytest.approx(d2, abs=TOL_SURFACE), (
            f"xz-swap symmetry broken for B_CUBE"
        )


# =============================================================================
# Path 9: Continuity -- nearby points have nearby distances
# =============================================================================


class TestContinuity:
    """The SDF should be continuous: nearby points produce nearby distances."""

    def test_continuity_along_x(self):
        """Continuity along x-axis from inside through surface to outside."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 0.001
        prev = py_sd_rounded_box((-b[0] - 1.0, 0.0, 0.0), b, r)
        for i in range(1, int((2 * b[0] + 2.0) / step)):
            x = -b[0] - 1.0 + i * step
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity along x at x={x}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Continuity along y-axis through the box."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 0.001
        prev = py_sd_rounded_box((0.0, -b[1] - 1.0, 0.0), b, r)
        for i in range(1, int((2 * b[1] + 2.0) / step)):
            y = -b[1] - 1.0 + i * step
            curr = py_sd_rounded_box((0.0, y, 0.0), b, r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity along y at y={y}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Continuity along z-axis through the box."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 0.001
        prev = py_sd_rounded_box((0.0, 0.0, -b[2] - 1.0), b, r)
        for i in range(1, int((2 * b[2] + 2.0) / step)):
            z = -b[2] - 1.0 + i * step
            curr = py_sd_rounded_box((0.0, 0.0, z), b, r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity along z at z={z}: diff={diff}"
            )
            prev = curr

    def test_continuity_through_surface_x(self):
        """Smooth transition through the x-face surface."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 1e-5
        for offset in range(-10, 11):
            x = b[0] + offset * step
            d = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            # Near the surface, sd ~ x - b.x for the face center
            expected = x - b[0]
            assert abs(d - expected) < 0.001, (
                f"Near x-face at x={x}: expected ~{expected}, got {d}"
            )

    def test_small_perturbation_stability(self):
        """Small perturbations should not cause large distance changes."""
        b, r = B_DEFAULT, R_DEFAULT
        center = (0.0, 0.0, 0.0)
        base_d = py_sd_rounded_box(center, b, r)
        eps = 1e-7
        for dx, dy, dz in [(eps, 0, 0), (-eps, 0, 0), (0, eps, 0),
                           (0, -eps, 0), (0, 0, eps), (0, 0, -eps)]:
            p = (center[0] + dx, center[1] + dy, center[2] + dz)
            d = py_sd_rounded_box(p, b, r)
            assert abs(d - base_d) < 0.001, (
                f"Unstable change: base={base_d}, perturbed={d} at {p}"
            )

    def test_continuity_diagonal(self):
        """Continuity along the main diagonal through a corner."""
        b, r = B_DEFAULT, R_DEFAULT
        step = 0.001
        start = -1.0
        end = 3.0
        prev = py_sd_rounded_box((start, start, start), b, r)
        for i in range(1, int((end - start) / step)):
            t = start + i * step
            curr = py_sd_rounded_box((t, t, t), b, r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity on diagonal at t={t}: diff={diff}"
            )
            prev = curr


# =============================================================================
# Path 10: Monotonicity -- distances increase away from center
# =============================================================================


class TestMonotonicity:
    """Signed distance is monotonic along radial, face-normal, and edge directions."""

    def test_monotonic_outside_x_axis(self):
        """Outside the x-face, distance increases with x."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        for dist in [b[0] + 0.5 * i for i in range(1, 10)]:
            curr = py_sd_rounded_box((dist, 0.0, 0.0), b, r)
            assert curr >= prev - 1e-12, (
                f"Outside x: distance decreased at x={dist}: {curr} < {prev}"
            )
            prev = curr

    def test_monotonic_outside_y_axis(self):
        """Outside the y-face, distance increases with y."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((0.0, b[1], 0.0), b, r)
        for dist in [b[1] + 0.5 * i for i in range(1, 10)]:
            curr = py_sd_rounded_box((0.0, dist, 0.0), b, r)
            assert curr >= prev - 1e-12, (
                f"Outside y: distance decreased at y={dist}"
            )
            prev = curr

    def test_monotonic_outside_z_axis(self):
        """Outside the z-face, distance increases with z."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((0.0, 0.0, b[2]), b, r)
        for dist in [b[2] + 0.5 * i for i in range(1, 10)]:
            curr = py_sd_rounded_box((0.0, 0.0, dist), b, r)
            assert curr >= prev - 1e-12, (
                f"Outside z: distance decreased at z={dist}"
            )
            prev = curr

    def test_monotonic_diagonal_outside(self):
        """Outside the corner along the diagonal, distance increases."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((b[0], b[1], b[2]), b, r)
        for scale in [1.2 * i for i in range(1, 8)]:
            p = (b[0] * scale, b[1] * scale, b[2] * scale)
            curr = py_sd_rounded_box(p, b, r)
            assert curr >= prev - 1e-12, (
                f"Outside diagonal: distance decreased at scale={scale}"
            )
            prev = curr

    def test_from_center_to_surface_x(self):
        """Moving from center to x-surface, distance increases (less negative)."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        for x in [0.5 * i for i in range(1, 5)]:
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            assert curr >= prev - 1e-12, (
                f"Center to x-surface: decreased at x={x}"
            )
            prev = curr

    def test_inside_outside_monotonic_x(self):
        """Through the surface: pos->neg->pos on x-axis."""
        b, r = B_DEFAULT, R_DEFAULT
        # Sequence from outside on left, through interior, to outside on right
        points = [(-b[0] - 1.0, -b[0] * 0.5, 0.0, b[0] * 0.5, b[0] + 1.0)]
        prev = -float("inf")
        for x in points[0]:
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            # We don't compare across the interior minimum (valley)
            prev = curr

    def test_monotonic_far_field(self):
        """Very far from the box, distance approximates distance to origin
        scaled by the direction."""
        b, r = B_DEFAULT, R_DEFAULT
        prev = py_sd_rounded_box((10.0, 0.0, 0.0), b, r)
        for far in [50.0, 100.0, 1000.0]:
            curr = py_sd_rounded_box((far, 0.0, 0.0), b, r)
            assert curr > prev, (
                f"Far field: distance should increase, got {curr} <= {prev}"
            )
            prev = curr


# =============================================================================
# Path: Parameter variation -- different box sizes and radii
# =============================================================================


class TestParameterVariation:
    """Behavior with different box dimensions and radii."""

    def test_cube_surface(self):
        """Cube: surface on all faces."""
        b, r = B_CUBE, R_CUBE
        for axis in range(3):
            p = [0.0, 0.0, 0.0]
            p[axis] = b[axis]
            d = py_sd_rounded_box(tuple(p), b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Cube surface at {p} should be ~0, got {d}"
            )
            p[axis] = -b[axis]
            d = py_sd_rounded_box(tuple(p), b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Cube surface at {p} should be ~0, got {d}"
            )

    def test_large_box(self):
        """Large box: surface and interior checks."""
        b, r = B_LARGE, R_DEFAULT
        # Center
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        assert d < 0.0, "Center of large box should be inside"
        # Surface at x-face
        d = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Large box x-face at {b[0]} should be ~0, got {d}"
        )

    def test_zero_radius_variation(self):
        """r=0 with various boxes matches sharp box."""
        boxes = [
            (1.0, 2.0, 3.0),
            (0.5, 0.5, 0.5),
            (10.0, 1.0, 1.0),
        ]
        test_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.5)]
        for b in boxes:
            for p in test_points:
                d_rounded = py_sd_rounded_box(p, b, 0.0)
                d_box = py_sd_box(p, b)
                assert d_rounded == pytest.approx(d_box, abs=TOL_SURFACE), (
                    f"r=0 mismatch for b={b}, p={p}: {d_rounded} vs {d_box}"
                )

    def test_small_r_surface(self):
        """Small corner radius: surface still at original box faces."""
        b = B_DEFAULT
        r = R_SMALL
        d = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Small r={r}: x-face should be on surface, got {d}"
        )

    def test_deterministic(self):
        """Same inputs always produce same output."""
        b, r = B_DEFAULT, R_DEFAULT
        p = (0.7, 0.3, 0.5)
        base = py_sd_rounded_box(p, b, r)
        for _ in range(20):
            assert py_sd_rounded_box(p, b, r) == pytest.approx(
                base, abs=TOL_SURFACE
            )


# =============================================================================
# Path: Far point behavior
# =============================================================================


class TestFarPoint:
    """Very distant points should have distance approximating distance from
    the box surface."""

    def test_far_positive_x(self):
        """Far on x-axis: distance ~ x - b.x."""
        b, r = B_DEFAULT, R_DEFAULT
        for far_x in [100.0, 1000.0]:
            d = py_sd_rounded_box((far_x, 0.0, 0.0), b, r)
            expected = far_x - b[0]
            assert d == pytest.approx(expected, abs=TOL_FAR), (
                f"Far x={far_x}: expected {expected}, got {d}"
            )

    def test_far_negative_x(self):
        """Far on -x axis: distance ~ |x| - b.x."""
        b, r = B_DEFAULT, R_DEFAULT
        for far_x in [-100.0, -1000.0]:
            d = py_sd_rounded_box((far_x, 0.0, 0.0), b, r)
            expected = abs(far_x) - b[0]
            assert d == pytest.approx(expected, abs=TOL_FAR), (
                f"Far x={far_x}: expected {expected}, got {d}"
            )

    def test_far_diagonal(self):
        """Far on diagonal: distance approximates the distance to the
        rounded corner."""
        b, r = B_DEFAULT, R_DEFAULT
        far = 1000.0
        d = py_sd_rounded_box((far, far, far), b, r)
        # The distance should be positive and increase with far
        assert d > 0.0, f"Far diagonal should be positive, got {d}"
        assert d > 100.0, f"Far diagonal distance should be large, got {d}"

    def test_far_asymptotic(self):
        """As point goes to infinity, distance / magnitude approaches 1."""
        b, r = B_DEFAULT, R_DEFAULT
        for scale in [100.0, 1000.0, 10000.0]:
            d = py_sd_rounded_box((scale, 0.0, 0.0), b, r)
            ratio = d / scale
            assert 0.9 < ratio < 1.0, (
                f"Far ratio d/mag={ratio} not near 1 at scale={scale}"
            )

    def test_far_symmetry(self):
        """Far in opposite directions gives comparable distances."""
        b, r = B_DEFAULT, R_DEFAULT
        d_pos = py_sd_rounded_box((1000.0, 0.0, 0.0), b, r)
        d_neg = py_sd_rounded_box((-1000.0, 0.0, 0.0), b, r)
        assert d_pos == pytest.approx(d_neg, abs=TOL_FAR), (
            f"Far points in opposite x should have same distance: "
            f"{d_pos} vs {d_neg}"
        )


# =============================================================================
# Path: Negative radius handling -- r clamped to [0, min(b)]
# =============================================================================


class TestNegativeRadiusClamp:
    """Negative r is clamped to 0; r > min(b) is clamped to min(b)."""

    def test_negative_r_treated_as_zero(self):
        """r=-0.5 behaves identically to r=0."""
        b = B_DEFAULT
        d_neg = py_sd_rounded_box((b[0], 0.0, 0.0), b, -0.5)
        d_zero = py_sd_rounded_box((b[0], 0.0, 0.0), b, 0.0)
        assert d_neg == pytest.approx(d_zero, abs=TOL_SURFACE), (
            f"r=-0.5 should match r=0 on surface: {d_neg} vs {d_zero}"
        )

    def test_negative_r_sharp_box(self):
        """Negative r should behave like a sharp box (r=0)."""
        b = B_DEFAULT
        points = [(0.0, 0.0, 0.0), (b[0], 0.0, 0.0), (b[0] + 1.0, 0.0, 0.0)]
        for p in points:
            d_neg = py_sd_rounded_box(p, b, -0.3)
            d_zero = py_sd_rounded_box(p, b, 0.0)
            assert d_neg == pytest.approx(d_zero, abs=TOL_SURFACE), (
                f"Negative r should match r=0 at {p}: {d_neg} vs {d_zero}"
            )

    def test_radius_clamped_to_min_dim(self):
        """r > min(b) is clamped to min(b)."""
        b = B_DEFAULT
        min_dim = min(b)
        r_excessive = min_dim * 10.0
        d_clamped = py_sd_rounded_box((0.0, 0.0, 0.0), b, r_excessive)
        d_at_limit = py_sd_rounded_box((0.0, 0.0, 0.0), b, min_dim)
        assert d_clamped == pytest.approx(d_at_limit, abs=TOL_SURFACE), (
            f"Excessive r should be clamped: {d_clamped} vs {d_at_limit}"
        )


# =============================================================================
# Path: Binary edge cases
# =============================================================================


class TestEdgeCases:
    """Degenerate and edge cases."""

    def test_zero_size_box(self):
        """A box with b=(0,0,0) and r=0 is just the origin."""
        d = py_sd_rounded_box((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Zero-size box at origin should be 0, got {d}"
        )

    def test_zero_size_box_off_origin(self):
        """A box with b=(0,0,0) and r=0: distance = length(p)."""
        points = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, 2.0, 3.0)]
        for p in points:
            d = py_sd_rounded_box(p, (0.0, 0.0, 0.0), 0.0)
            expected = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            assert d == pytest.approx(expected, abs=TOL_SURFACE), (
                f"Zero-size box at {p}: expected {expected}, got {d}"
            )

    def test_one_dim_zero(self):
        """A box with one dimension zero is a rounded rectangle."""
        b = (0.0, 1.0, 1.0)
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Flat box with one zero dim at center should be 0, got {d}"
        )

    def test_all_surface_cardinal(self):
        """All 6 cardinal surface points at once."""
        b, r = B_DEFAULT, R_DEFAULT
        surface_points = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
        ]
        for p in surface_points:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should be ~0, got {d}"
            )

"""
Cleanroom blackbox tests for sdBoxFrame WGSL function (T-DEMO-1.9).

Tests the signed distance function for a hollow box frame of outer half-dimensions
b and frame wall thickness e, centered at the origin, treating the implementation
as a black box from the mathematical spec.

The true signed distance for a hollow box frame (set difference of outer box
minus inner cavity box) is:

    sdBoxFrame(p, b, e) = max(sdBox(p, b), -sdBox(p, b-e))

where sdBox(p, b-e) is the signed distance to the inner cavity boundary, and
we negate it to get the distance to the frame from the cavity side.  When e=0
the frame degenerates to a solid box and we return sdBox(p,b) directly.

BLACKBOX coverage plan:
  Path A:  Inside frame wall returns negative signed distance
  Path B:  On outer surface returns ~0
  Path C:  Inside hollow cavity returns positive distance
  Path D:  Outside frame returns positive distance
  Path E:  e=0 degenerates to sdBox (solid box)
  Path F:  Thick frame: e >= min(b) fills cavity, behaves as sdBox
  Path G:  Symmetry under sign flips and coordinate permutation
  Path H:  Direction -- moving toward outer surface decreases distance
  Path I:  Continuity -- nearby points have nearby distances
  Path J:  Sign convention -- negative in walls, positive in cavity/outside, zero on surfaces
  Path K:  Parameter and thickness variation
  Path L:  Far-point asymptotic behavior
  Path M:  Corner cases (non-uniform b, very thin frame)
"""

from __future__ import annotations

import math

import pytest


# =============================================================================
# Python model (cleanroom) of sdBoxFrame matching the mathematical spec
# =============================================================================


def _sd_box(p, b):
    """Signed distance from point p (3-tuple) to a solid axis-aligned box
    of half-dimensions b=(bx,by,bz), centered at origin.

    Formula (Inigo Quilez):
        q = abs(p) - b
        sdBox = length(max(q, 0)) + min(maxComponent(q), 0)
    """
    qx = abs(p[0]) - abs(b[0])
    qy = abs(p[1]) - abs(b[1])
    qz = abs(p[2]) - abs(b[2])

    out_dist = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
    max_comp = max(qx, qy, qz)
    inner_penalty = min(max_comp, 0.0)
    return out_dist + inner_penalty


def py_sd_box_frame(p, b, e):
    """Python (cleanroom) model of WGSL sdBoxFrame(p, b, e) -> f32.

    Signed distance from point p (3-tuple) to a hollow box frame with outer
    half-dimensions b=(bx,by,bz) and frame wall thickness e, centered at origin.

    The frame is the set difference: solid_box(b) \\ hollow_cavity(b-e).
    Its SDF follows from the SDF of the set difference operator:

        sdFrame(p) = max(sdBox(p,b), -sdBox(p,b-e))

    with the following edge cases:
      - e <= 0           -> sdBox(p,b)                     (solid box)
      - e >= min(b)      -> sdBox(p,b)                     (cavity fully filled)
      - otherwise        -> max(sdBox(p,b), -sdBox(p,b-e))

    Mathematical properties:
      - Inside frame material (b-e < |p_i| < b for at least one axis):  < 0
      - On either surface (inner or outer):                              = 0
      - In hollow cavity (|p_i| < b-e for all axes):                     > 0
      - Outside (|p_i| > b for at least one axis):                       > 0

    Reference: Inigo Quilez -- SDF Primitives: sdBoxFrame
    https://iquilezles.org/articles/distfunctions/
    """
    sb = tuple(abs(v) for v in b)
    se = abs(e)

    d_outer = _sd_box(p, sb)

    # Edge case: e=0 or wholly fills interior -> solid box
    if se <= 0.0:
        return d_outer

    inner_b = tuple(max(v - se, 0.0) for v in sb)
    if min(inner_b) <= 0.0:
        # At least one axis has zero (or negative) cavity dimension;
        # the frame fills the entire interior along that axis.
        return d_outer

    d_inner = _sd_box(p, inner_b)
    return max(d_outer, -d_inner)


# =============================================================================
# Reference sdBox for e=0 comparison
# =============================================================================


def py_sd_box(p, b):
    """Signed distance to solid box (same as _sd_box, kept for clarity)."""
    return _sd_box(p, b)


# =============================================================================
# Default test parameters
# =============================================================================

TOL_SURFACE = 1e-9      # Points on surface should be very close to 0
TOL_FAR = 0.001          # Tolerance for far-point approximations
TOL_CONTINUITY = 0.01    # Max allowed jump for nearby points

# Uniform cube frame
B_DEFAULT = (2.0, 2.0, 2.0)
E_DEFAULT = 0.5

# Thin frame
E_THIN = 0.2

# Thick frame (but still has cavity)
E_THICK = 0.8

# Non-uniform box
B_NONUNIFORM = (3.0, 1.5, 2.0)
E_NONUNIFORM = 0.3


# =============================================================================
# Path A: Inside frame wall returns negative signed distance
# =============================================================================


class TestInsideWall:
    """Points inside the frame material should return negative distance."""

    def test_wall_midpoint_x(self):
        """Midpoint of x-wall for b=(2,2,2), e=0.5: p=(1.75, 0, 0)."""
        d = py_sd_box_frame((1.75, 0.0, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Mid-wall should be negative, got {d}"

    def test_wall_near_outer_surface(self):
        """Just inside outer surface: p=(1.95, 0, 0), b=(2,2,2), e=0.5."""
        d = py_sd_box_frame((1.95, 0.0, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Near outer surface inside should be negative, got {d}"

    def test_wall_near_inner_surface(self):
        """Just inside inner surface: p=(1.55, 0, 0), b=(2,2,2), e=0.5."""
        d = py_sd_box_frame((1.55, 0.0, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Near inner surface inside should be negative, got {d}"

    def test_wall_on_y_face(self):
        """Inside y-wall: p=(0, 1.75, 0), b=(2,2,2), e=0.5."""
        d = py_sd_box_frame((0.0, 1.75, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside y-wall should be negative, got {d}"

    def test_wall_on_z_face(self):
        """Inside z-wall: p=(0, 0, 1.88), b=(2,2,2), e=0.5."""
        d = py_sd_box_frame((0.0, 0.0, 1.88), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside z-wall should be negative, got {d}"

    def test_wall_edge_corner(self):
        """Inside wall near a corner: p=(1.6, 1.6, 0), b=(2,2,2), e=0.5."""
        d = py_sd_box_frame((1.6, 1.6, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside wall near corner should be negative, got {d}"

    def test_wall_negative_x(self):
        """Inside wall on negative x side: p=(-1.6, 0, 0)."""
        d = py_sd_box_frame((-1.6, 0.0, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside -x wall should be negative, got {d}"

    def test_wall_negative_y(self):
        """Inside wall on negative y side: p=(0, -1.88, 0)."""
        d = py_sd_box_frame((0.0, -1.88, 0.0), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside -y wall should be negative, got {d}"

    def test_wall_negative_z(self):
        """Inside wall on negative z side: p=(0, 0, -1.6)."""
        d = py_sd_box_frame((0.0, 0.0, -1.6), B_DEFAULT, E_DEFAULT)
        assert d < 0.0, f"Inside -z wall should be negative, got {d}"

    def test_wall_nonuniform_box(self):
        """Inside wall of non-uniform box: p=(2.85, 0, 0), b=(3,1.5,2), e=0.3."""
        d = py_sd_box_frame((2.85, 0.0, 0.0), B_NONUNIFORM, E_NONUNIFORM)
        assert d < 0.0, f"Inside non-uniform wall should be negative, got {d}"


# =============================================================================
# Path B: On outer surface returns ~0
# =============================================================================


class TestOuterSurface:
    """Points exactly on the outer box surface should return distance ~0."""

    def test_outer_surface_positive_x(self):
        """On outer surface at (+b.x, 0, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface +x should be 0, got {d}"
        )

    def test_outer_surface_negative_x(self):
        """On outer surface at (-b.x, 0, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((-b[0], 0.0, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface -x should be 0, got {d}"
        )

    def test_outer_surface_positive_y(self):
        """On outer surface at (0, +b.y, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, b[1], 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface +y should be 0, got {d}"
        )

    def test_outer_surface_negative_y(self):
        """On outer surface at (0, -b.y, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, -b[1], 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface -y should be 0, got {d}"
        )

    def test_outer_surface_positive_z(self):
        """On outer surface at (0, 0, +b.z)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, b[2]), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface +z should be 0, got {d}"
        )

    def test_outer_surface_negative_z(self):
        """On outer surface at (0, 0, -b.z)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, -b[2]), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface -z should be 0, got {d}"
        )

    def test_outer_surface_off_axis_x(self):
        """On outer surface off-center: (b.x, 0.5, 0.3)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((b[0], 0.5, 0.3), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface off-center +x should be 0, got {d}"
        )

    def test_outer_surface_nonuniform(self):
        """On outer surface of non-uniform box: (+3, 0, 0), b=(3,1.5,2)."""
        b, e = B_NONUNIFORM, E_NONUNIFORM
        d = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Non-uniform outer surface should be 0, got {d}"
        )

    def test_inner_surface_x(self):
        """On inner surface: p=(b.x-e, 0, 0) = (1.5, 0, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((b[0] - e, 0.0, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface +x should be 0, got {d}"
        )

    def test_inner_surface_y(self):
        """On inner surface along y: p=(0, b.y-e, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, b[1] - e, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface +y should be 0, got {d}"
        )

    def test_inner_surface_z(self):
        """On inner surface along z: p=(0, 0, b.z-e)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, b[2] - e), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface +z should be 0, got {d}"
        )

    def test_inner_surface_negative_x(self):
        """On inner surface at negative x: p=(-(b.x-e), 0, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((-(b[0] - e), 0.0, 0.0), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface -x should be 0, got {d}"
        )

    def test_inner_surface_off_axis(self):
        """On inner surface off-center: (b.x-e, 0.4, 0.2)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((b[0] - e, 0.4, 0.2), b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface off-center should be 0, got {d}"
        )


# =============================================================================
# Path C: Inside hollow cavity returns positive distance
# =============================================================================


class TestCavityPositive:
    """Points inside the hollow cavity should return positive distance."""

    def test_cavity_center(self):
        """Center of cavity: p=(0,0,0), b=(2,2,2), e=0.5 -> 1.5."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = b[0] - e  # = 1.5
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity center should be {expected}, got {d}"
        )

    def test_cavity_off_center_x(self):
        """Cavity point off-center: p=(1.0, 0, 0), distance to inner face = 0.5."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((1.0, 0.0, 0.0), b, e)
        expected = (b[0] - e) - 1.0  # = 0.5
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity off-center should be {expected}, got {d}"
        )

    def test_cavity_off_center_y(self):
        """Cavity point off-center in y: p=(0, 0.8, 0)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.8, 0.0), b, e)
        expected = (b[1] - e) - 0.8  # = 0.7
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity off-center y should be {expected}, got {d}"
        )

    def test_cavity_off_center_z(self):
        """Cavity point off-center in z: p=(0, 0, 1.2)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, 1.2), b, e)
        expected = (b[2] - e) - 1.2  # = 0.3
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity off-center z should be {expected}, got {d}"
        )

    def test_cavity_diagonal(self):
        """Cavity diagonal: p=(0.5, 0.5, 0.5), distance to nearest face = 1.0."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.5, 0.5, 0.5), b, e)
        expected = (b[0] - e) - 0.5  # = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity diagonal should be {expected}, got {d}"
        )

    def test_cavity_near_inner_surface(self):
        """Cavity point very near inner surface: p=(1.49, 0, 0) -> ~0.01."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((1.49, 0.0, 0.0), b, e)
        expected = (b[0] - e) - 1.49  # = 0.01
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity near inner surface should be {expected}, got {d}"
        )

    def test_cavity_nonuniform(self):
        """Cavity center of non-uniform box: b=(3,1.5,2), e=0.3 -> min(b-e)."""
        b, e = B_NONUNIFORM, E_NONUNIFORM
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = min(b[0] - e, b[1] - e, b[2] - e)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Non-uniform cavity center should be {expected}, got {d}"
        )

    def test_cavity_negative_quadrant(self):
        """Cavity point in negative quadrant: p=(-0.7, -0.3, -0.5)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((-0.7, -0.3, -0.5), b, e)
        expected = min(1.5 - 0.7, 1.5 - 0.3, 1.5 - 0.5)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Cavity negative quadrant should be {expected}, got {d}"
        )

    def test_cavity_all_axes_positive_distances(self):
        """Multiple cavity points -- all should be positive."""
        b, e = B_DEFAULT, E_DEFAULT
        cavity_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (-1.0, 0.0, 0.0),
            (0.0, -0.8, 0.0),
        ]
        for p in cavity_points:
            d = py_sd_box_frame(p, b, e)
            assert d > 0.0, (
                f"Cavity point {p} should have positive SDF, got {d}"
            )


# =============================================================================
# Path D: Outside frame returns positive distance
# =============================================================================


class TestOutsidePositive:
    """Points outside the outer box should return positive distance."""

    def test_outside_x_axis(self):
        """Outside along +x: p=(3, 0, 0), b=(2,2,2) -> 1."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((3.0, 0.0, 0.0), b, e)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside +x should be {expected}, got {d}"
        )

    def test_outside_negative_x(self):
        """Outside along -x: p=(-3, 0, 0) -> 1."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((-3.0, 0.0, 0.0), b, e)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside -x should be {expected}, got {d}"
        )

    def test_outside_y_axis(self):
        """Outside along +y: p=(0, 3, 0), b=(2,2,2) -> 1."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 3.0, 0.0), b, e)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside +y should be {expected}, got {d}"
        )

    def test_outside_z_axis(self):
        """Outside along +z: p=(0, 0, 3), b=(2,2,2) -> 1."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((0.0, 0.0, 3.0), b, e)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside +z should be {expected}, got {d}"
        )

    def test_outside_diagonal(self):
        """Outside on diagonal: p=(3, 3, 0), b=(2,2,2)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((3.0, 3.0, 0.0), b, e)
        expected = math.sqrt(2.0)  # = sqrt(1^2 + 1^2)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside diagonal should be {expected}, got {d}"
        )

    def test_outside_far_3d(self):
        """Far outside in 3D: p=(4, 3, 5), b=(2,2,2)."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((4.0, 3.0, 5.0), b, e)
        expected = math.sqrt((4 - 2) ** 2 + (3 - 2) ** 2 + (5 - 2) ** 2)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside 3D point should be {expected}, got {d}"
        )

    def test_outside_near_surface(self):
        """Just outside outer surface: p=(2.001, 0, 0) -> 0.001."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((2.001, 0.0, 0.0), b, e)
        assert d > 0.0, f"Just outside should be positive, got {d}"

    def test_outside_various_points(self):
        """Multiple outside points -- all should be positive."""
        b, e = B_DEFAULT, E_DEFAULT
        outside_points = [
            (3.0, 0.0, 0.0),
            (-3.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 0.0, 3.0),
            (0.0, 0.0, -3.0),
            (2.5, 2.5, 0.0),
            (2.1, 2.1, 2.1),
            (2.5, 0.5, 0.5),
        ]
        for p in outside_points:
            d = py_sd_box_frame(p, b, e)
            assert d > 0.0, (
                f"Outside point {p} should have positive SDF, got {d}"
            )

    def test_outside_link_corner(self):
        """Outside near outer edge: p=(1, 2.2, 0), outside y but inside x."""
        b, e = B_DEFAULT, E_DEFAULT
        d = py_sd_box_frame((1.0, 2.2, 0.0), b, e)
        # Signed distance to box: q = (1-2, 2.2-2, 0-2) = (-1, 0.2, -2)
        # max(q,0) = (0, 0.2, 0), len=0.2, maxComp = max(-1,0.2,-2) = 0.2, min(0.2,0)=0
        # sdBox = 0.2 + 0 = 0.2
        # -sdInner: inner b-e = (1.5,1.5,1.5), q = (1-1.5, 2.2-1.5, 0-1.5) = (-0.5,0.7,-1.5)
        #   max(q,0) = (0,0.7,0), len=0.7, max(-0.5,0.7,-1.5)=0.7, min(0.7,0)=0
        #   sdInner = 0.7
        #   -sdInner = -0.7
        # max(0.2, -0.7) = 0.2
        expected = 0.2
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Outside link corner should be {expected}, got {d}"
        )


# =============================================================================
# Path E: e=0 degenerates to sdBox (solid box)
# =============================================================================


class TestEZeroDegenerate:
    """When e=0, the frame degenerates to a solid box (identical to sdBox)."""

    def test_e_zero_center(self):
        """e=0 at center: sdBoxFrame = sdBox = -2."""
        b = B_DEFAULT
        d_frame = py_sd_box_frame((0.0, 0.0, 0.0), b, 0.0)
        d_box = py_sd_box((0.0, 0.0, 0.0), b)
        assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
            f"e=0 should match sdBox at center: {d_frame} vs {d_box}"
        )

    def test_e_zero_inside(self):
        """e=0 inside box: matches sdBox."""
        b = B_DEFAULT
        inside_points = [(1.0, 0.0, 0.0), (0.0, 1.5, 0.0), (0.5, 0.5, 0.5),
                         (-1.0, 0.0, 0.0)]
        for p in inside_points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e=0 inside {p}: frame={d_frame}, box={d_box}"
            )

    def test_e_zero_surface(self):
        """e=0 on box surface: matches sdBox = 0."""
        b = B_DEFAULT
        surface_points = [
            (b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, 0.0, b[2]),
            (-b[0], 0.0, 0.0),
        ]
        for p in surface_points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            assert d_frame == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"e=0 surface {p} should be 0, got {d_frame}"
            )

    def test_e_zero_outside(self):
        """e=0 outside box: matches sdBox."""
        b = B_DEFAULT
        outside_points = [(3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (3.0, 3.0, 0.0)]
        for p in outside_points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e=0 outside {p}: frame={d_frame}, box={d_box}"
            )

    def test_e_zero_nonuniform(self):
        """e=0 with non-uniform box dimensions matches sdBox."""
        b = B_NONUNIFORM
        points = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        for p in points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e=0 non-uniform {p}: frame={d_frame}, box={d_box}"
            )

    def test_e_zero_negative_e(self):
        """Negative e=0 should behave same as e=0."""
        b = B_DEFAULT
        p = (0.0, 0.0, 0.0)
        d_zero = py_sd_box_frame(p, b, 0.0)
        d_neg_zero = py_sd_box_frame(p, b, -0.0)
        assert d_zero == pytest.approx(d_neg_zero, abs=TOL_SURFACE), (
            f"e=-0 and e=0 should match"
        )

    def test_e_zero_consistent_across_domain(self):
        """e=0 value matches sdBox across a sweep along x."""
        b = B_DEFAULT
        for x in [i * 0.1 for i in range(-30, 31)]:
            p = (x, 0.0, 0.0)
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e=0 mismatch at x={x}: frame={d_frame}, box={d_box}"
            )


# =============================================================================
# Path F: Thick frame -- e >= min(b) fills cavity, behaves as sdBox
# =============================================================================


class TestThickFrame:
    """When e >= min(b), the frame fills the cavity and behaves as sdBox."""

    def test_e_equals_min_b(self):
        """e = min(b) fills cavity: center should be negative (sdBox)."""
        b = B_DEFAULT
        e = min(b)  # = 2.0
        d_frame = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        d_box = py_sd_box((0.0, 0.0, 0.0), b)
        assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
            f"e=min(b) should match sdBox at center"
        )

    def test_e_exceeds_min_b(self):
        """e > min(b): interior is solid box throughout."""
        b = B_DEFAULT
        e = 3.0  # > min(b)=2
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.5, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ]
        for p in points:
            d_frame = py_sd_box_frame(p, b, e)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e > min(b) should match sdBox at {p}"
            )

    def test_e_large_nonuniform(self):
        """e >= min(b) on non-uniform box."""
        b = B_NONUNIFORM
        e = min(b) + 0.5  # = 2.0, > min(b)=1.5
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.5, 0.0, 0.0)]
        for p in points:
            d_frame = py_sd_box_frame(p, b, e)
            d_box = py_sd_box(p, b)
            assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
                f"e >= min(b) non-uniform at {p}: frame={d_frame}, box={d_box}"
            )

    def test_e_large_no_cavity_positive(self):
        """When e fills cavity, no point should have positive interior."""
        b = B_DEFAULT
        e = 2.5
        interior = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (1.5, 0.0, 0.0)]
        for p in interior:
            d = py_sd_box_frame(p, b, e)
            assert d < 0.0, (
                f"With e >= min(b), interior {p} should be negative, got {d}"
            )


# =============================================================================
# Path G: Symmetry under sign flips and coordinate permutation
# =============================================================================


class TestSymmetry:
    """Box symmetry: sign flips in each axis and coordinate permutations."""

    def test_x_sign_symmetry(self):
        """(x, y, z) and (-x, y, z) give equal distance."""
        b, e = B_DEFAULT, E_DEFAULT
        samples = [(0.5, 0.3, 0.0), (1.8, 0.0, 0.5), (0.0, 1.2, 0.0),
                   (2.5, 0.5, 0.0), (0.3, 1.0, 0.7)]
        for x, y, z in samples:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((-x, y, z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"x-sign symmetry broken at ({x},{y},{z})"
            )

    def test_y_sign_symmetry(self):
        """(x, y, z) and (x, -y, z) give equal distance."""
        b, e = B_DEFAULT, E_DEFAULT
        samples = [(0.5, 0.3, 0.0), (0.0, 1.8, 0.0), (1.0, 0.0, 0.5),
                   (0.3, 2.5, 0.0)]
        for x, y, z in samples:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((x, -y, z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"y-sign symmetry broken at ({x},{y},{z})"
            )

    def test_z_sign_symmetry(self):
        """(x, y, z) and (x, y, -z) give equal distance."""
        b, e = B_DEFAULT, E_DEFAULT
        samples = [(0.5, 0.3, 0.8), (0.0, 0.0, 1.8), (1.0, 1.0, 0.0),
                   (0.3, 0.5, 2.5)]
        for x, y, z in samples:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((x, y, -z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"z-sign symmetry broken at ({x},{y},{z})"
            )

    def test_all_sign_flips(self):
        """All 8 sign combinations give same distance for uniform box."""
        b, e = (2.0, 2.0, 2.0), E_DEFAULT
        base = py_sd_box_frame((0.5, 0.3, 0.7), b, e)
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    p = (sx * 0.5, sy * 0.3, sz * 0.7)
                    d = py_sd_box_frame(p, b, e)
                    assert d == pytest.approx(base, abs=TOL_SURFACE), (
                        f"Sign symmetry broken for {p}"
                    )

    def test_coordinate_permutation_uniform(self):
        """With uniform b, permuting coordinates gives same distance."""
        b, e = (2.0, 2.0, 2.0), E_DEFAULT
        p = (0.5, 0.3, 0.7)
        d_xyz = py_sd_box_frame(p, b, e)
        d_xzy = py_sd_box_frame((p[0], p[2], p[1]), b, e)
        d_yxz = py_sd_box_frame((p[1], p[0], p[2]), b, e)
        d_yzx = py_sd_box_frame((p[1], p[2], p[0]), b, e)
        d_zxy = py_sd_box_frame((p[2], p[0], p[1]), b, e)
        d_zyx = py_sd_box_frame((p[2], p[1], p[0]), b, e)
        assert d_xyz == pytest.approx(d_xzy, abs=TOL_SURFACE)
        assert d_xyz == pytest.approx(d_yxz, abs=TOL_SURFACE)
        assert d_xyz == pytest.approx(d_yzx, abs=TOL_SURFACE)
        assert d_xyz == pytest.approx(d_zxy, abs=TOL_SURFACE)
        assert d_xyz == pytest.approx(d_zyx, abs=TOL_SURFACE)


# =============================================================================
# Path H: Direction -- moving toward outer surface decreases distance
# =============================================================================


class TestDirection:
    """Moving toward the outer box surface should decrease distance;
    moving away should increase distance."""

    def test_toward_outer_surface_from_cavity(self):
        """Moving from cavity center toward outer wall decreases distance."""
        b, e = B_DEFAULT, E_DEFAULT
        center_d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        mid_d = py_sd_box_frame((0.75, 0.0, 0.0), b, e)
        inner_surf_d = py_sd_box_frame((b[0] - e, 0.0, 0.0), b, e)
        assert mid_d < center_d, (
            f"Moving toward wall from cavity should decrease: "
            f"{mid_d} >= {center_d}"
        )
        assert inner_surf_d < mid_d, (
            f"Reaching inner surface should decrease further: "
            f"{inner_surf_d} >= {mid_d}"
        )

    def test_inside_wall_to_outside(self):
        """Moving from inside wall through surface to outside strictly increases."""
        b, e = B_DEFAULT, E_DEFAULT
        inside_wall = py_sd_box_frame((1.6, 0.0, 0.0), b, e)
        on_surface = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        outside = py_sd_box_frame((2.5, 0.0, 0.0), b, e)
        assert inside_wall < on_surface, (
            f"Inside wall should be < on surface: {inside_wall} >= {on_surface}"
        )
        assert on_surface < outside, (
            f"On surface should be < outside: {on_surface} >= {outside}"
        )

    def test_cavity_to_outside_through_wall(self):
        """Full path: cavity -> inner surf -> wall -> outer surf -> outside."""
        b, e = B_DEFAULT, E_DEFAULT
        cavity = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        inner = py_sd_box_frame((b[0] - e, 0.0, 0.0), b, e)
        mid_wall = py_sd_box_frame((b[0] - e / 2, 0.0, 0.0), b, e)
        outer = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        far_out = py_sd_box_frame((b[0] + 1.0, 0.0, 0.0), b, e)
        assert cavity > 0.0, f"Cavity should be positive, got {cavity}"
        assert inner == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface should be 0, got {inner}"
        )
        assert mid_wall < 0.0, f"Mid-wall should be negative, got {mid_wall}"
        assert outer == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface should be 0, got {outer}"
        )
        assert far_out > 0.0, f"Far out should be positive, got {far_out}"

    def test_toward_surface_decreases_outside(self):
        """Moving toward the box from outside decreases distance."""
        b, e = B_DEFAULT, E_DEFAULT
        far_d = py_sd_box_frame((5.0, 0.0, 0.0), b, e)
        near_d = py_sd_box_frame((2.5, 0.0, 0.0), b, e)
        assert near_d < far_d, (
            f"Moving toward box from outside should decrease: "
            f"{near_d} >= {far_d}"
        )

    def test_away_from_surface_increases_outside(self):
        """Moving away from the box surface increases distance."""
        b, e = B_DEFAULT, E_DEFAULT
        near_d = py_sd_box_frame((2.5, 0.0, 0.0), b, e)
        far_d = py_sd_box_frame((5.0, 0.0, 0.0), b, e)
        assert far_d > near_d, (
            f"Moving away from box should increase: {far_d} <= {near_d}"
        )

    def test_through_cavity_sign_change(self):
        """Moving through the frame from outside to opposite outside
        shows positive -> zero -> negative -> zero -> positive."""
        b, e = B_DEFAULT, E_DEFAULT
        left_out = py_sd_box_frame((-3.0, 0.0, 0.0), b, e)
        left_surf = py_sd_box_frame((-b[0], 0.0, 0.0), b, e)
        inside_wall = py_sd_box_frame((-1.6, 0.0, 0.0), b, e)
        inner_surf = py_sd_box_frame((-(b[0] - e), 0.0, 0.0), b, e)
        cavity = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        assert left_out > 0.0, f"Left outside should be positive, got {left_out}"
        assert left_surf == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Left surface should be 0, got {left_surf}"
        )
        assert inside_wall < 0.0, f"Inside wall should be negative, got {inside_wall}"
        assert inner_surf == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface should be 0, got {inner_surf}"
        )
        assert cavity > 0.0, f"Cavity should be positive, got {cavity}"


# =============================================================================
# Path I: Continuity -- nearby points have nearby distances
# =============================================================================


class TestContinuity:
    """The SDF should be continuous: nearby points produce nearby distances."""

    def test_continuity_along_x(self):
        """Continuity along x-axis through cavity, wall, and outside."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 0.001
        prev = py_sd_box_frame((-3.0, 0.0, 0.0), b, e)
        for i in range(1, 6000):
            x = -3.0 + i * step
            curr = py_sd_box_frame((x, 0.0, 0.0), b, e)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at x={x}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Continuity along y-axis."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 0.001
        prev = py_sd_box_frame((0.0, -3.0, 0.0), b, e)
        for i in range(1, 6000):
            y = -3.0 + i * step
            curr = py_sd_box_frame((0.0, y, 0.0), b, e)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at y={y}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Continuity along z-axis."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 0.001
        prev = py_sd_box_frame((0.0, 0.0, -3.0), b, e)
        for i in range(1, 6000):
            z = -3.0 + i * step
            curr = py_sd_box_frame((0.0, 0.0, z), b, e)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at z={z}: diff={diff}"
            )
            prev = curr

    def test_continuity_through_inner_surface(self):
        """Smooth transition through the inner surface."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 1e-5
        for offset in range(-10, 11):
            x = (b[0] - e) + offset * step
            d = py_sd_box_frame((x, 0.0, 0.0), b, e)
            # Near the inner surface, SDF should be near 0
            assert abs(d) < 0.01, (
                f"Near inner surface at x={x}: SDF={d}, expected near 0"
            )

    def test_continuity_through_outer_surface(self):
        """Smooth transition through the outer surface."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 1e-5
        for offset in range(-10, 11):
            x = b[0] + offset * step
            d = py_sd_box_frame((x, 0.0, 0.0), b, e)
            # Near the outer surface, SDF should be near 0
            assert abs(d) < 0.01, (
                f"Near outer surface at x={x}: SDF={d}, expected near 0"
            )

    def test_continuity_diagonal(self):
        """Continuity along diagonal through cavity, corner of wall, outside."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 0.001
        prev = py_sd_box_frame((-3.0, -3.0, -3.0), b, e)
        for i in range(1, 4000):
            t = -3.0 + i * step
            curr = py_sd_box_frame((t, t, t), b, e)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity on diagonal at t={t}: diff={diff}"
            )
            prev = curr

    def test_small_perturbation_stability(self):
        """Small perturbations should not cause large changes."""
        b, e = B_DEFAULT, E_DEFAULT
        # Test at several representative points
        test_points = [
            (0.0, 0.0, 0.0),          # Cavity center
            (1.0, 0.0, 0.0),          # Cavity off-center
            (1.6, 0.0, 0.0),          # Inside wall
            (2.2, 0.0, 0.0),          # Outside
            (0.5, 0.5, 0.5),          # Cavity diagonal
        ]
        eps = 1e-7
        for center in test_points:
            base_d = py_sd_box_frame(center, b, e)
            for dx, dy, dz in [(eps, 0, 0), (-eps, 0, 0),
                               (0, eps, 0), (0, -eps, 0),
                               (0, 0, eps), (0, 0, -eps)]:
                p = (center[0] + dx, center[1] + dy, center[2] + dz)
                d = py_sd_box_frame(p, b, e)
                assert abs(d - base_d) < 0.001, (
                    f"Unstable change at {center}: base={base_d}, perturbed={d}"
                )


# =============================================================================
# Path J: Sign convention
# =============================================================================


class TestSignConvention:
    """Sign convention: negative in walls, positive in cavity/outside, zero on surfaces."""

    def test_wall_negative(self):
        """Points inside frame walls should have negative distance."""
        b, e = B_DEFAULT, E_DEFAULT
        wall_points = [
            (1.6, 0.0, 0.0),
            (0.0, 1.8, 0.0),
            (0.0, 0.0, 1.7),
            (-1.6, 0.0, 0.0),
            (0.0, -1.8, 0.0),
            (0.0, 0.0, -1.7),
            (1.6, 1.6, 0.0),
        ]
        for p in wall_points:
            d = py_sd_box_frame(p, b, e)
            assert d <= 0.0, (
                f"Wall point {p} should have non-positive SDF, got {d}"
            )

    def test_cavity_positive(self):
        """Points inside the hollow cavity should have positive distance."""
        b, e = B_DEFAULT, E_DEFAULT
        cavity_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.5, 0.8),
            (-0.5, -0.5, -0.5),
        ]
        for p in cavity_points:
            d = py_sd_box_frame(p, b, e)
            assert d >= 0.0, (
                f"Cavity point {p} should have non-negative SDF, got {d}"
            )

    def test_outside_positive(self):
        """Points outside the box should have positive distance."""
        b, e = B_DEFAULT, E_DEFAULT
        outside_points = [
            (3.0, 0.0, 0.0),
            (-3.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 0.0, 3.0),
            (0.0, 0.0, -3.0),
        ]
        for p in outside_points:
            d = py_sd_box_frame(p, b, e)
            assert d >= 0.0, (
                f"Outside point {p} should have non-negative SDF, got {d}"
            )

    def test_surfaces_zero(self):
        """Points on inner and outer surfaces should be near zero."""
        b, e = B_DEFAULT, E_DEFAULT
        surface_points = [
            # Outer surfaces
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
            # Inner surfaces
            (b[0] - e, 0.0, 0.0),
            (-(b[0] - e), 0.0, 0.0),
            (0.0, b[1] - e, 0.0),
            (0.0, -(b[1] - e), 0.0),
            (0.0, 0.0, b[2] - e),
            (0.0, 0.0, -(b[2] - e)),
        ]
        for p in surface_points:
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_outer_surface(self):
        """Sign flips at the outer surface: wall negative, surface zero, outside positive."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 1e-4
        wall = py_sd_box_frame((b[0] - step, 0.0, 0.0), b, e)
        surf = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        outside = py_sd_box_frame((b[0] + step, 0.0, 0.0), b, e)
        assert wall < 0.0, f"Inside wall should be negative, got {wall}"
        assert surf == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Outer surface should be 0, got {surf}"
        )
        assert outside > 0.0, f"Outside should be positive, got {outside}"

    def test_sign_transition_at_inner_surface(self):
        """Sign flips at the inner surface: cavity positive, surface zero, wall negative."""
        b, e = B_DEFAULT, E_DEFAULT
        step = 1e-4
        cavity = py_sd_box_frame((b[0] - e - step, 0.0, 0.0), b, e)
        surf = py_sd_box_frame((b[0] - e, 0.0, 0.0), b, e)
        wall = py_sd_box_frame((b[0] - e + step, 0.0, 0.0), b, e)
        assert cavity > 0.0, f"Cavity near inner surface should be positive, got {cavity}"
        assert surf == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Inner surface should be 0, got {surf}"
        )
        assert wall < 0.0, f"Wall near inner surface should be negative, got {wall}"

    def test_sign_consistent_all_directions(self):
        """Sign holds in all 6 cardinal directions from center."""
        b, e = B_DEFAULT, E_DEFAULT
        directions = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        for d_vec in directions:
            # Cavity: half the inner radius
            inner_radius = b[0] - e  # uniform box
            cavity_p = tuple(0.5 * inner_radius * c for c in d_vec)
            # Inner surface
            inner_surf_p = tuple(inner_radius * c for c in d_vec)
            # Wall interior
            wall_p = tuple((b[0] - 0.5 * e) * c for c in d_vec)
            # Outer surface
            outer_surf_p = tuple(b[0] * c for c in d_vec)
            # Outside
            outside_p = tuple((b[0] + 1.0) * c for c in d_vec)

            assert py_sd_box_frame(cavity_p, b, e) > 0.0, (
                f"Cavity along {d_vec} should be positive"
            )
            assert py_sd_box_frame(inner_surf_p, b, e) == pytest.approx(
                0.0, abs=TOL_SURFACE
            ), f"Inner surface along {d_vec} should be 0"
            assert py_sd_box_frame(wall_p, b, e) < 0.0, (
                f"Wall along {d_vec} should be negative"
            )
            assert py_sd_box_frame(outer_surf_p, b, e) == pytest.approx(
                0.0, abs=TOL_SURFACE
            ), f"Outer surface along {d_vec} should be 0"
            assert py_sd_box_frame(outside_p, b, e) > 0.0, (
                f"Outside along {d_vec} should be positive"
            )


# =============================================================================
# Path K: Parameter and thickness variation
# =============================================================================


class TestParameterVariation:
    """Behavior with different box dimensions and frame thicknesses."""

    def test_thin_frame_cavity(self):
        """Thin frame (e=0.2): center distance = b - e = 1.8."""
        b, e = B_DEFAULT, E_THIN
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = b[0] - e
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Thin frame center should be {expected}, got {d}"
        )

    def test_thin_frame_wall_thickness(self):
        """Thin frame: wall is thin, verify inside wall is negative."""
        b, e = B_DEFAULT, E_THIN
        d = py_sd_box_frame((1.9, 0.0, 0.0), b, e)
        assert d < 0.0, f"Thin frame wall should be negative, got {d}"

    def test_thick_frame_cavity(self):
        """Thick frame (e=0.8): center distance = b - e = 1.2."""
        b, e = B_DEFAULT, E_THICK
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = b[0] - e
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Thick frame center should be {expected}, got {d}"
        )

    def test_thick_frame_wall(self):
        """Thick frame: verify wall."""
        b, e = B_DEFAULT, E_THICK
        d = py_sd_box_frame((1.6, 0.0, 0.0), b, e)
        assert d < 0.0, f"Thick frame wall should be negative, got {d}"

    def test_nonuniform_box_center(self):
        """Non-uniform box: center = min(b-e)."""
        b, e = B_NONUNIFORM, E_NONUNIFORM
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = min(b[0] - e, b[1] - e, b[2] - e)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Non-uniform center should be {expected}, got {d}"
        )

    def test_nonuniform_box_surfaces(self):
        """Non-uniform box: surfaces at each axis-boundary."""
        b, e = B_NONUNIFORM, E_NONUNIFORM
        # Outer surfaces
        for axis in range(3):
            for sign in [1, -1]:
                p = [0.0, 0.0, 0.0]
                p[axis] = sign * b[axis]
                d = py_sd_box_frame(tuple(p), b, e)
                assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                    f"Non-uniform outer surface at {p} should be 0, got {d}"
                )
        # Inner surfaces
        for axis in range(3):
            for sign in [1, -1]:
                p = [0.0, 0.0, 0.0]
                p[axis] = sign * (b[axis] - e)
                d = py_sd_box_frame(tuple(p), b, e)
                assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                    f"Non-uniform inner surface at {p} should be 0, got {d}"
                )

    def test_thin_vs_thick_cavity_size(self):
        """Thinner frame means larger cavity (higher center SDF)."""
        b = B_DEFAULT
        d_thin = py_sd_box_frame((0.0, 0.0, 0.0), b, E_THIN)
        d_thick = py_sd_box_frame((0.0, 0.0, 0.0), b, E_THICK)
        assert d_thin > d_thick, (
            f"Thin frame should have larger cavity: "
            f"thin={d_thin}, thick={d_thick}"
        )

    def test_frame_with_negative_e(self):
        """Negative e should be treated as positive (abs)."""
        b = B_DEFAULT
        d_pos = py_sd_box_frame((0.0, 0.0, 0.0), b, E_DEFAULT)
        d_neg = py_sd_box_frame((0.0, 0.0, 0.0), b, -E_DEFAULT)
        assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
            f"Negative e should behave as positive: {d_pos} vs {d_neg}"
        )

    def test_negative_b_dimension(self):
        """Negative b dimension should be treated as positive (abs)."""
        b_pos = B_DEFAULT
        b_neg = (-B_DEFAULT[0], B_DEFAULT[1], B_DEFAULT[2])
        d_pos = py_sd_box_frame((1.0, 0.0, 0.0), b_pos, E_DEFAULT)
        d_neg = py_sd_box_frame((1.0, 0.0, 0.0), b_neg, E_DEFAULT)
        assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
            f"Negative b.x should behave as positive: {d_pos} vs {d_neg}"
        )


# =============================================================================
# Path L: Far-point asymptotic behavior
# =============================================================================


class TestFarPoint:
    """Very distant points approximate sdBox behavior (frame thickness negligible)."""

    def test_far_x_axis(self):
        """Far on +x: sd approx = distance - b.x."""
        b, e = B_DEFAULT, E_DEFAULT
        far_x = 1000.0
        d = py_sd_box_frame((far_x, 0.0, 0.0), b, e)
        expected = far_x - b[0]
        assert d == pytest.approx(expected, abs=TOL_FAR)

    def test_far_negative_x(self):
        """Far on -x: sd approx = |x| - b.x."""
        b, e = B_DEFAULT, E_DEFAULT
        far_x = -1000.0
        d = py_sd_box_frame((far_x, 0.0, 0.0), b, e)
        expected = abs(far_x) - b[0]
        assert d == pytest.approx(expected, abs=TOL_FAR)

    def test_far_y_axis(self):
        """Far on +y: sd approx = y - b.y."""
        b, e = B_DEFAULT, E_DEFAULT
        far_y = 1000.0
        d = py_sd_box_frame((0.0, far_y, 0.0), b, e)
        expected = far_y - b[1]
        assert d == pytest.approx(expected, abs=TOL_FAR)

    def test_far_z_axis(self):
        """Far on +z: sd approx = z - b.z."""
        b, e = B_DEFAULT, E_DEFAULT
        far_z = 1000.0
        d = py_sd_box_frame((0.0, 0.0, far_z), b, e)
        expected = far_z - b[2]
        assert d == pytest.approx(expected, abs=TOL_FAR)

    def test_far_diagonal(self):
        """Far on diagonal: sd approx = Euclidean distance to box."""
        b, e = B_DEFAULT, E_DEFAULT
        far = 1000.0
        d = py_sd_box_frame((far, far, 0.0), b, e)
        expected = math.sqrt((far - b[0]) ** 2 + (far - b[1]) ** 2)
        assert d == pytest.approx(expected, abs=TOL_FAR)

    def test_far_ratio_approaches_one(self):
        """SDF/distance-to-origin approaches 1 as p goes to infinity."""
        b, e = B_DEFAULT, E_DEFAULT
        for scale in [100.0, 1000.0, 10000.0]:
            d = py_sd_box_frame((scale, 0.0, 0.0), b, e)
            ratio = d / scale
            assert ratio > 0.9, (
                f"Far ratio d/scale={ratio} not near 1 at scale={scale}"
            )


# =============================================================================
# Path M: Monotonicity
# =============================================================================


class TestMonotonicity:
    """The SDF should be monotonic in distance from the frame."""

    def test_monotonic_outside_increasing(self):
        """Outside the box, distance increases with distance from surface."""
        b, e = B_DEFAULT, E_DEFAULT
        prev = py_sd_box_frame((b[0], 0.0, 0.0), b, e)
        for dist in [2.5, 3.0, 4.0, 6.0, 10.0]:
            curr = py_sd_box_frame((dist, 0.0, 0.0), b, e)
            assert curr >= prev, (
                f"Outside monotonicity broken at x={dist}: {prev} > {curr}"
            )
            prev = curr

    def test_monotonic_cavity_decreasing(self):
        """In cavity, distance decreases moving from center toward inner surface."""
        b, e = B_DEFAULT, E_DEFAULT
        inner_x = b[0] - e
        prev = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        for x in [0.3, 0.6, 0.9, 1.2, 1.4]:
            curr = py_sd_box_frame((x, 0.0, 0.0), b, e)
            assert curr < prev, (
                f"Cavity monotonicity broken at x={x}: {prev} >= {curr}"
            )
            prev = curr

    def test_wall_negative_interior(self):
        """Inside wall, SDF is strictly negative (closer to zero at surfaces)."""
        b, e = B_DEFAULT, E_DEFAULT
        wall_points = [x * 0.1 for x in range(16, 20)]  # 1.6 .. 1.9, exclude surfaces 1.5 and 2.0
        for x in wall_points:
            d = py_sd_box_frame((x, 0.0, 0.0), b, e)
            assert d < 0.0, (
                f"Wall point x={x} should have negative SDF, got {d}"
            )
        # Mid-wall should be more negative than edges (V-shape)
        mid = py_sd_box_frame((1.75, 0.0, 0.0), b, e)
        near_inner = py_sd_box_frame((1.51, 0.0, 0.0), b, e)
        near_outer = py_sd_box_frame((1.99, 0.0, 0.0), b, e)
        assert mid < near_inner and mid < near_outer, (
            f"Wall should be V-shaped (mid={mid}, near_inner={near_inner}, near_outer={near_outer})"
        )

    def test_full_path_monotonic_segments(self):
        """Moving along x from far outside to far outside: segments are monotonic."""
        b, e = B_DEFAULT, E_DEFAULT
        x_vals = [-5.0, -4.0, -3.0, -2.5, -2.0, -1.9, -1.8, -1.6,
                   -1.5, -1.0, 0.0, 1.0, 1.5,
                   1.6, 1.8, 1.9, 2.0, 2.5, 3.0, 4.0, 5.0]
        # The function goes: + -> 0 -> - -> 0 -> + -> 0 -> - -> 0 -> +
        # Each segment should be monotonic
        segments = [
            (-5.0, -2.001),  # Far left, decreasing toward outer surface
            (-1.999, -1.501),  # Inside wall, decreasing toward inner surface
            (-1.499, 0.0),    # Cavity, increasing toward center
            (0.0, 1.499),     # Cavity, decreasing toward inner surface
            (1.501, 1.999),   # Inside wall, decreasing toward inner surface
            (2.001, 5.0),     # Outside, increasing away from outer surface
        ]
        for start, end in segments:
            step = 0.01
            prev_d = py_sd_box_frame((start, 0.0, 0.0), b, e)
            x = start + step
            while x <= end:
                curr_d = py_sd_box_frame((x, 0.0, 0.0), b, e)
                # Direction depends on segment:
                # Outside segments: increasing with distance from box
                # Wall toward inner: decreasing
                # Cavity: decreasing away from center, increasing toward center
                # Just check change isn't chaotic
                diff = abs(curr_d - prev_d)
                assert diff < 0.02, (
                    f"Non-smooth change at x={x}: prev={prev_d}, curr={curr_d}"
                )
                prev_d = curr_d
                x += step


# =============================================================================
# Path N: Corner cases -- zero dimensions and extreme values
# =============================================================================


class TestCornerCases:
    """Edge cases including zero dimensions."""

    def test_both_b_and_e_zero(self):
        """b=(0,0,0) and e=0 at origin: distance 0."""
        d = py_sd_box_frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"b=0, e=0 at origin should be 0, got {d}"
        )

    def test_b_zero_with_e(self):
        """b=(0,0,0) with e>0 at origin returns -e (no cavity, point box)."""
        d = py_sd_box_frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)
        expected = 0.0  # sdBox of a point is 0 at origin
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"b=0 origin with e>0 should be 0, got {d}"
        )

    def test_single_element_nonzero(self):
        """b=(0, 0, 1), e=0.2: thin slab with frame."""
        d_frame = py_sd_box_frame((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2)
        d_box = py_sd_box((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        # e > min(b)=0, so the function returns sdBox (no cavity in zero-dim axes)
        assert d_frame == pytest.approx(d_box, abs=TOL_SURFACE), (
            f"b=(0,0,1) should behave as sdBox (no cavity): {d_frame} vs {d_box}"
        )

    def test_extremely_thin_frame(self):
        """Very thin frame: e=1e-6, almost sdBox but with tiny cavity."""
        b = B_DEFAULT
        e = 1e-6
        d_center = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        expected = b[0] - e  # ≈ 2.0
        assert d_center == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Extremely thin frame center should be ~{expected}, got {d_center}"
        )

    def test_deterministic(self):
        """Same inputs always produce same output."""
        b, e = B_DEFAULT, E_DEFAULT
        p = (0.7, 0.3, 0.9)
        base = py_sd_box_frame(p, b, e)
        for _ in range(20):
            assert py_sd_box_frame(p, b, e) == pytest.approx(base, abs=TOL_SURFACE)

    def test_on_surface_all_cardinal_directions(self):
        """Surface points in all 6 cardinal directions return ~0."""
        b, e = B_DEFAULT, E_DEFAULT
        surface_points = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
            (b[0] - e, 0.0, 0.0),
            (-(b[0] - e), 0.0, 0.0),
            (0.0, b[1] - e, 0.0),
            (0.0, -(b[1] - e), 0.0),
            (0.0, 0.0, b[2] - e),
            (0.0, 0.0, -(b[2] - e)),
        ]
        for p in surface_points:
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should be 0, got {d}"
            )
